import streamlit as st
import fitz  # PyMuPDF
import io
import re
from quantulum3 import parser
import pandas as pd
import json
import hashlib
from pathlib import Path
import base64
from classifier_esrs import classify_esrs
import uuid
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(layout="wide")

page = st.sidebar.selectbox("📄 Page", ["ESRS Display", "Backend Debugger"])
st.title("ESG Extractor")

st.markdown(
    """
    <style>
    #context-box {
        position: fixed;
        right: 2rem;
        top: 6rem;
        width: 30%;
        max-height: 80vh;
        overflow-y: auto;
        background: #111;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
        z-index: 9999;
        color: white;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ---------- Helpers ----------
def get_pdf_hash(pdf_bytes):
    return hashlib.md5(pdf_bytes).hexdigest()

def extract_text_by_page(doc):
    lines = []
    for page_number in range(len(doc)):
        text = doc[page_number].get_text()
        for line in text.split("\n"):
            lines.append({"text": line.strip(), "page": page_number + 1})
    return lines

def extract_full_text_by_page(doc):
    return [page.get_text() for page in doc]

def get_context_snippet(page_text, sentence, window=200):
    text = page_text.strip()
    sentence = sentence.strip()
    match = re.search(re.escape(sentence[:40]), text, re.IGNORECASE)
    if match:
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        snippet = text[start:end]
        return snippet.replace(sentence, f"**{sentence}**")
    idx = text.lower().find(sentence[:30].lower())
    if idx != -1:
        start = max(0, idx - window)
        end = min(len(text), idx + len(sentence) + window)
        return text[start:end].replace(sentence, f"**{sentence}**")
    return f"[\u274c Context not found]\n\n{text[:1000]}"

def tag_relevance(result):
    context = result["sentence"].lower()
    value = result["value"]
    surface = result["surface"]
    score = 0
    if "employee" in context: score += 1
    if "training" in context: score += 1
    if "emission" in context: score += 1
    if "percent" in context or "%" in surface: score += 1
    if "scope" in context: score -= 1
    result["score"] = score
    result["relevance"] = "relevant" if score >= 1 else "uncertain"
    return result

def extract_number_phrases(doc):
    results = []
    for line_obj in extract_text_by_page(doc):
        line = line_obj["text"]
        page = line_obj["page"]
        if not line or len(line.split()) < 3:
            continue
        for q in parser.parse(line):
            result = {
                "page": page,
                "sentence": line,
                "value": q.value,
                "unit": q.unit.name if q.unit else None,
                "surface": q.surface,
            }
            tag_relevance(result)
            results.append(result)
    return results

def classify_all(rows, page_texts):
    classified = []
    for row in rows:
        page_text = page_texts[row["page"] - 1] if 1 <= row["page"] <= len(page_texts) else ""
        result = classify_esrs(row["sentence"])
        if result["category"] == "Uncategorized" or result["metric"] == "unknown_metric":
            snippet = get_context_snippet(page_text, row["sentence"])
            result = classify_esrs(snippet)
        row.update(result)
        row["id"] = f"{row['page']}_{hash(row['sentence']) % 100000}"
        row["feedback"] = None
        classified.append(row)
    return classified

def save_dataframe(df, path):
    df.to_json(path, orient="records", indent=2)

def load_dataframe(path):
    return pd.read_json(path)

def make_random_id():
    return uuid.uuid4().hex[:10]  # 10-char unique ID
# ---------- Streamlit Logic ----------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

df_placeholder = st.empty() 
if uploaded_file:
    try:
        pdf_bytes = uploaded_file.read()
        pdf_hash = get_pdf_hash(pdf_bytes)
        data_path = Path(f"data_{pdf_hash}.json")

        if data_path.exists():
            df = load_dataframe(data_path)
            # st.success(" Loaded cached data for this PDF.")

            # Load PDF + extract page text for context view
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            st.session_state["page_texts"] = extract_full_text_by_page(doc)
        else:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            page_texts = extract_full_text_by_page(doc)
            st.session_state["page_texts"] = page_texts

            raw_rows = extract_number_phrases(doc)
            df = pd.DataFrame(classify_all(raw_rows, page_texts))
            df["id"] = [make_random_id() for _ in range(len(df))]
            save_dataframe(df, data_path)
            st.success("PDF processed and classified.")

        def render_table(df_subset, label):
            st.markdown(f"### {label}")
            if df_subset.empty:
                st.info("No values found.")
                return

            rows_per_page = 20
            total_pages = (len(df_subset) - 1) // rows_per_page + 1
            page_key = f"page_{label}"

            if page_key not in st.session_state:
                st.session_state[page_key] = 1

            st.markdown(f"**Page {st.session_state[page_key]} / {total_pages} for {label}**")

            current_page = st.number_input(
                "",
                min_value=1,
                max_value=total_pages,
                value=st.session_state[page_key],
                key=page_key,
                label_visibility="collapsed"
            )

            start_idx = (current_page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            current_rows = df_subset.iloc[start_idx:end_idx]

            for i, row in current_rows.iterrows():
                row_id = row["id"]
                cols = st.columns([1, 0.5, 1, 1.2, 0.5, 2, 1])

                if cols[0].button(f"{row['page']}", key=f"pgbtn_{label}_{row_id}"):
                    st.session_state["preview_row"] = row.to_dict()

                cols[1].markdown(f"`{row['surface']}`")
                new_value = cols[2].text_input("value", str(row["value"]), key=f"valfix_{row_id}", label_visibility="collapsed")
                unit_options = ["dimensionless", "percentage", "EUR", "USD", "hour", "tons", "other"]
                new_unit = cols[3].selectbox("unit", unit_options, index=unit_options.index(row["unit"]) if row["unit"] in unit_options else 0, key=f"unitfix_{row_id}", label_visibility="collapsed")
                cols[4].markdown(f"{row['score']}")
                cols[5].markdown(row["sentence"][:150])

                with cols[6]:
                    if st.button("up", key=f"up_{row_id}"):
                        df.loc[df["id"] == row_id, "relevance"] = "relevant"
                        df.loc[df["id"] == row_id, "feedback"] = "relevant"
                        save_dataframe(df, data_path)
                        st.rerun()

                    if st.button("down", key=f"down_{row_id}"):
                        df.loc[df["id"] == row_id, "relevance"] = "uncertain"
                        df.loc[df["id"] == row_id, "feedback"] = "not_relevant"
                        save_dataframe(df, data_path)
                        st.rerun()

                    if st.button("save", key=f"save_{row_id}"):
                        try:
                            df.loc[df["id"] == row_id, "value"] = float(new_value)
                        except ValueError:
                            pass
                        df.loc[df["id"] == row_id, "unit"] = new_unit
                        save_dataframe(df, data_path)
                        st.rerun()


        df = load_dataframe(data_path)

        if "id" not in df.columns:
            st.error("❌ This file is missing unique IDs. Please reprocess the PDF.")
            st.stop()

        df["__safe__"] = ""  # Permanent dummy column at end

        if page == "Backend Debugger":
            if st.button("🔁 Retry Classification (with Context Fallback)"):
                st.info("Reclassifying sentences with unknown category or metric...")

                doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
                page_texts = extract_full_text_by_page(doc)

                def retry_classification(row):
                    if row.get("metric") == "unknown_metric" or row.get("category") == "Uncategorized":
                        snippet = get_context_snippet(page_texts[row["page"] - 1], row["sentence"])
                        new_result = classify_esrs(snippet)
                        row.update(new_result)
                    return row

                df = df.apply(retry_classification, axis=1)
                save_dataframe(df, data_path)
                st.success("✅ Reclassification complete.")
                st.rerun()
            left, right = st.columns([2,1])

            cols_to_hide = ["id", "feedback"]
            visible_cols = [col for col in df.columns if col not in cols_to_hide]
            df_clean = df[visible_cols].copy()
            df_placeholder.dataframe(df_clean)

            relevant_df = df[df["relevance"] == "relevant"].reset_index(drop=True)
            uncertain_df = df[df["relevance"] == "uncertain"].reset_index(drop=True)

            with left:
                render_table(relevant_df, " Relevant")
                render_table(uncertain_df, "Uncertain")

            with right:
                if "preview_row" in st.session_state:
                    row = st.session_state["preview_row"]
                    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
                    page_texts = extract_full_text_by_page(doc)
                    context = get_context_snippet(page_texts[row["page"] - 1], row["sentence"])

                    if st.button("Close Preview", key="close_preview"):
                        st.session_state.pop("preview_row", None)
                        st.rerun()

                    st.markdown(f"""
                    <div id='context-box'>
                    <h4>Page {row['page']}</h4>
                    <pre>{context}</pre>
                    </div>
                    """, unsafe_allow_html=True)

        elif page == "ESRS Display":
            import textwrap

            st.title("🧾 ESRS KPI Dashboard")

            left, right = st.columns([2, 1])

            with left:
                # 1. Filter relevant data
                relevant_df = df[df["relevance"] == "relevant"].copy()
                categories = sorted(relevant_df["category"].dropna().unique())

                # 2. Category picker
                selected_cat = st.selectbox("🗂️ Select ESRS Category", categories)
                filtered_df = relevant_df[relevant_df["category"] == selected_cat]
                metrics = sorted(filtered_df["metric"].dropna().unique())

                # 3. Show glowing tiles for each metric
                st.markdown("## ✨ Highlighted Metrics")

                for metric in metrics:
                    st.markdown(f"#### 🔹 {metric.replace('_', ' ').title()}")
                    rows = filtered_df[filtered_df["metric"] == metric].head(3)  # limit tiles per metric

                    cols = st.columns(3)
                    for i, (_, row) in enumerate(rows.iterrows()):
                        with cols[i % 3]:
                            value = float(row["value"]) if str(row["value"]).replace('.', '', 1).isdigit() else None
                            unit = row["unit"] or ""
                            sentence = textwrap.shorten(row["sentence"], width=80, placeholder="…")

                            bar = f'<progress value="{value}" max="100" style="width:100%;"></progress>' if value is not None and unit == "percentage" else ""

                            st.markdown(f"""
                            <div style="
                                padding: 1em;
                                border-radius: 12px;
                                background: #1e1e1e;
                                color: white;
                                text-align: center;
                                box-shadow: 0 2px 10px rgba(0,0,0,0.4);
                                margin-bottom: 1em;
                            ">
                                <div style="font-size: 1.5em; font-weight: bold;">
                                    {row['value']} {unit}
                                </div>
                                <div style="font-size: 0.9em; margin-top: 0.5em;">
                                    {sentence}
                                </div>
                                <div style="font-size: 0.8em; color: #aaa; margin-top: 0.4em;">
                                    📄 Page {row['page']}
                                </div>
                                {bar}
                            </div>
                            """, unsafe_allow_html=True)

                # 4. Show category summary
                st.markdown("## 📊 Overall ESRS Coverage")

                cat_counts = relevant_df["category"].value_counts().reset_index()
                cat_counts.columns = ["category", "count"]

                for _, row in cat_counts.iterrows():
                    st.markdown(f"""
                    <div style="padding:1em;margin:0.5em 0;background:#222;color:#fff;
                                border-left:6px solid #0f0;border-radius:4px;">
                        <h4 style="margin:0;">{row['category']}</h4>
                        <p style="margin:0.2em 0 0 0;">🟢 {row['count']} metrics extracted</p>
                    </div>
                    """, unsafe_allow_html=True)
            with right:
                st.markdown("### 📊 Metrics Overview")

                # Prep clean data
                perc_df = relevant_df[relevant_df["unit"] == "percentage"].copy()
                perc_df["value"] = pd.to_numeric(perc_df["value"], errors="coerce")

                # 1. Average % value per category
                st.markdown("#### 📈 Avg % per Category")
                avg_by_cat = perc_df.groupby("category")["value"].mean().reset_index()
                st.bar_chart(avg_by_cat.set_index("category"))

                # 2. Count of metrics per ESRS category
                st.markdown("#### 📊 Count by Category")
                cat_counts = relevant_df["category"].value_counts()
                st.bar_chart(cat_counts)

                # 3. Metric breakdown donut chart (Plotly)
                import plotly.express as px

                top_metrics = relevant_df["metric"].value_counts().head(6).reset_index()
                top_metrics.columns = ["Metric", "Count"]

                fig = px.pie(top_metrics, values="Count", names="Metric", hole=0.4,
                             title="Top 6 Metric Types")
                st.plotly_chart(fig, use_container_width=True)

                # 4. Optional radar chart placeholder (later)
                st.markdown("#### 🕸️ Balance Radar (Coming Soon...)")
                st.caption("Compare normalized values across ESG categories.")

            # st.title("🧾 ESRS KPIs")
            #
            # relevant_df = df[df["relevance"] == "relevant"].copy()
            # categories = sorted(relevant_df["category"].dropna().unique())
            #
            # selected_cat = st.selectbox("🗂️ Select ESRS Category", categories)
            #
            # filtered_df = relevant_df[relevant_df["category"] == selected_cat]
            # metrics = sorted(filtered_df["metric"].dropna().unique())
            #
            # for metric in metrics:
            #     st.markdown(f"#### 🔹 {metric.replace('_', ' ').title()}")
            #     rows = filtered_df[filtered_df["metric"] == metric]
            #
            #     table_df = rows[["value", "unit", "page", "sentence"]].copy()
            #     table_df["Value"] = table_df["value"].astype(str) + " " + table_df["unit"].fillna("")
            #     table_df["Page"] = table_df["page"]
            #     table_df["Sentence"] = table_df["sentence"]
            #     table_df = table_df[["Value", "Page", "Sentence"]]
            #
            #     gb = GridOptionsBuilder.from_dataframe(table_df)
            #     gb.configure_column("Value", width=60)
            #     gb.configure_column("Page", width=50)
            #     gb.configure_column(
            #         "Sentence",
            #         flex=1,
            #         minWidth=300,
            #         wrapText=True,
            #         autoHeight=True,
            #         tooltipField="Sentence",
            #         suppressSizeToFit=True  # <- this is key!
            #     )
            #     gb.configure_grid_options(domLayout='normal')  # more performant
            #
            #     AgGrid(
            #         table_df,
            #         gridOptions=gb.build(),
            #         height=400,
            #         fit_columns_on_grid_load=False,
            #         enable_enterprise_modules=False
            #     )

    except Exception as e:
        st.error(f"\u274c Error processing PDF: {e}")
