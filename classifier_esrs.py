import spacy
from typing import Optional

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# --- Rule-based metric detection ---
METRIC_CATALOG = {
    "employee_diversity_ratio": [
        "gender", "male", "female", "woman", "women", "underrepresented",
        "minority", "diversity", "inclusion", "equality"
    ],
    "employee_training_hours": [
        "training", "hours", "learning", "development", "education", "professional"
    ],
    "esg_kpi_weighting": [
        "weight", "kpi", "incentive", "remuneration", "bonus"
    ],
    "emissions_reduction_target": [
        "target", "reduction", "reduce", "goal", "emissions"
    ],
    "emissions_change": [
        "decrease", "reduction", "increase", "emissions", "fell", "rose", "change"
    ],
    "board_independence_ratio": [
        "board", "director", "independent", "independence", "executive", "shareholder"
    ],
    "energy_savings": [
        "saving", "consumption", "energy", "electricity", "recovery", "efficient", "efficiency"
    ],
    "product_transparency_documents": [
        "hpd", "hpds", "epd", "declaration", "health product", "environmental product"
    ],
    "geographic_coverage": [
        "countries", "global", "regions", "international", "footprint", "operations"
    ],
    "health_safety_audit_coverage": [
        "audit", "health", "safety", "contractor", "coverage", "units", "ohsms", "hse"
    ],
    "child_labor_policy_minimum_age": [
        "minimum age", "child", "under the age", "youth", "hazardous work", "15", "18", "minor"
    ],
    "compliance_incidents_total": [
        "compliance", "violation", "incident", "report", "breach", "issue"
    ],
    "compliance_reporting_channel_ratio": [
        "compliance line", "hotline", "received through", "reporting channel"
    ],
    "employee_turnover_count": [
    "resigned", "dismissed", "termination", "left the company", "turnover", "resign", "terminated"
    ],
    "workplace_fatalities": [
        "fatality", "fatalities", "death", "died", "work-related", "accident", "incident"
    ]

}

# --- ESRS category detection ---
ESRS_RULES = {
    "E1: Climate change": [
        "emission", "carbon", "co2", "ghg", "climate", "greenhouse",
        "energy", "consumption", "renewable", "scope"
    ],
    "E2: Pollution": [
        "pollution", "waste", "toxic", "chemical", "hazardous", "pollutant"
    ],
    "E3: Water and marine resources": [
        "water", "marine", "ocean", "aquatic", "discharge", "wastewater"
    ],
    "E4: Biodiversity and ecosystems": [
        "biodiversity", "ecosystem", "habitat", "species", "nature", "land"
    ],
    "E5: Resource use and circular economy": [
        "recycle", "reuse", "efficiency", "material", "lifespan", "circular", "resource"
    ],
    "S1: Own workforce": [
        "employee", "gender", "training", "diversity", "health", "safety", "union",
        "workforce", "wage", "headcount", "iifr"
    ],
    "S2: Workers in the value chain": [
        "supplier", "contractor", "chain", "outsourced"
    ],
    "S3: Affected communities": [
        "community", "stakeholder", "indigenous", "land", "rights"
    ],
    "S4: Consumers and end-users": [
        "customer", "consumer", "product", "safety", "complaint", "privacy",
        "hpd", "epd", "declaration", "transparency"
    ],
    "G1: Business conduct": [
        "ethic", "corruption", "bribery", "governance", "compliance", "audit",
        "tax", "board", "independent", "independence", "corporation",
        "whistleblowing", "risk", "conduct", "policy", "lobbying", "report"
    ],
    "Meta: ESG strategy / KPI weighting": [
        "incentive", "bonus", "remuneration", "weighting", "performance-based",
        "executive", "kpi", "non-financial", "esg-linked", "target-setting"
    ]
}

# --- Optional semantic fallback ---
try:
    from sentence_transformers import SentenceTransformer, util
    import torch

    _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    SEMANTIC_METRICS = {
        "material_reuse_ratio": "percentage of materials that are reusable, recyclable, or recovered in the product lifecycle",
        "taxonomy_alignment_ratio": "percentage of capital or operating expenditure aligned with EU taxonomy for sustainable activities",
        "product_transparency_documents": "sustainability reporting documents like HPDs or EPDs, number of Health Product Declarations, green product disclosures",
        "geographic_coverage": "number of countries or regions where the company operates, including manufacturing or service presence",
        "health_safety_audit_coverage": "percentage of company units or sites covered by a health and safety management system audit scheme",
        "child_labor_policy_minimum_age": "minimum legal working age defined in company policies or code of conduct",
        "compliance_incidents_total": "total number of compliance reports, violations, or incidents reported to the company",
        "employee_turnover_count": "number of employees who resigned or were dismissed from the company in a given period",

        "workplace_fatalities": "number of employees who died due to work-related accidents or conditions",
        "employee_diversity_ratio": "gender breakdown or diversity in workforce",
        "employee_training_hours": "hours spent on employee training or development",
        "esg_kpi_weighting": "percentage of ESG KPIs in executive performance bonuses",
        "emissions_reduction_target": "targeted reduction in emissions",
        "emissions_change": "change in greenhouse gas emissions",
        "energy_savings": "percentage of energy saved through efficiency measures",
        "board_independence_ratio": "share of independent board members",
    }

    _metric_names = list(SEMANTIC_METRICS.keys())
    _metric_texts = list(SEMANTIC_METRICS.values())
    _metric_embeds = _semantic_model.encode(_metric_texts, convert_to_tensor=True)

    def classify_metric_semantic(text: str, threshold: float = 0.45) -> str:
        embedding = _semantic_model.encode(text, convert_to_tensor=True)
        sims = util.cos_sim(embedding, _metric_embeds)[0]

        best_idx = int(torch.argmax(sims))
        score = float(sims[best_idx])
        return _metric_names[best_idx] if score >= threshold else "unknown_metric"

except Exception as e:
    print("Semantic classification unavailable:", e)
    classify_metric_semantic = None

# --- Main classifier ---
def classify_esrs(sentence: str) -> dict:
    doc = nlp(sentence)
    lemmas = set(token.lemma_.lower() for token in doc)
    text_lower = sentence.lower()

    # Step 1: Smart entity-based shortcut
    for ent in doc.ents:
        if ent.label_ == "PERCENT":
            head = ent.root.head
            if head and head.pos_ in {"NOUN", "ADJ", "PROPN"}:
                head_lemma = head.lemma_.lower()
                if head_lemma in ["weight", "kpi", "incentive", "remuneration", "bonus"]:
                    return {
                        "category": "Meta: ESG strategy / KPI weighting",
                        "metric": "esg_kpi_weighting",
                        "breakdown_label": None
                    }
                if head_lemma in ["emission", "carbon", "co2", "ghg", "consumption", "energy"]:
                    return {
                        "category": "E1: Climate change",
                        "metric": "emissions_change",
                        "breakdown_label": None
                    }

    # Step 2: ESRS category match
    category = "Uncategorized"
    for cat, keywords in ESRS_RULES.items():
        if any(k in lemmas or k in text_lower for k in keywords):
            category = cat
            break

    # Step 3: Metric match
    metric = "unknown_metric"
    breakdown_label = None

    for metric_name, keywords in METRIC_CATALOG.items():
        if any(k in text_lower for k in keywords):
            metric = metric_name
            if metric == "employee_diversity_ratio":
                if "female" in text_lower or "women" in text_lower:
                    breakdown_label = "female"
                elif "underrepresented" in text_lower:
                    breakdown_label = "underrepresented"
                elif "male" in text_lower:
                    breakdown_label = "male"
            break

    # Step 4: Semantic fallback
    if metric == "unknown_metric" and classify_metric_semantic:
        metric = classify_metric_semantic(sentence)

    return {
        "category": category,
        "metric": metric,
        "breakdown_label": breakdown_label
    }
