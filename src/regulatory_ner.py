"""Objective 7: Regulatory Constraint Extraction (LLM/NER).

Extracts environmental constraints, permissible limits, substances, and compliance
obligations from regulatory text documents (e.g., CPCB/SPCB hazardous waste guidelines).

Implementation note: extraction is rule-based (keyword lists + regular expressions),
not a trained or fine-tuned NER model, and nothing here crawls documents. The saved
`models/regulatory_ner.joblib` only stores the rules extracted from
`data/processed/regulation_text.jsonl`.
"""
import json
import re
from pathlib import Path
import joblib
import pandas as pd


class RegulatoryEntityExtractor:
    """Extracts substances, thresholds, units, and regulatory directives from unstructured text."""
    
    SUBSTANCES = [
        "fly ash", "blast furnace slag", "steel slag", "fgd gypsum", "rice husk",
        "bagasse", "waste paper", "spent acid", "used oil", "textile sludge",
        "plastic scrap", "mill scale"
    ]
    
    PROPERTIES = {
        "heavy metal": "heavy_metals_ppm",
        "moisture": "moisture_pct",
        "calorific": "calorific_mj_kg",
        "storage": "storage_days_max",
        "stored": "storage_days_max",
        "ph": "ph"
    }

    def __init__(self):
        pass

    def parse_text(self, text):
        """Extracts structured entities and constraints from a single regulatory clause."""
        text_lower = text.lower()
        
        # 1. Extract Substance
        found_substance = None
        for sub in self.SUBSTANCES:
            if sub in text_lower:
                found_substance = sub
                break
                
        # 2. Extract Threshold & Unit
        threshold_match = re.search(r'(\d+(?:\.\d+)?)\s*(ppm|%|days|mj/kg|mg/l|inr/t)', text_lower)
        threshold_val = None
        unit_val = None
        if threshold_match:
            threshold_val = float(threshold_match.group(1))
            unit_val = threshold_match.group(2)
        else:
            num_match = re.search(r'(\d+(?:\.\d+)?)', text_lower)
            if num_match:
                threshold_val = float(num_match.group(1))

        # 3. Extract Property Target
        found_property = "general_limit"
        for prop_key, prop_name in self.PROPERTIES.items():
            if prop_key in text_lower:
                found_property = prop_name
                break
                
        # 4. Extract Obligation Directive
        obligation = "RESTRICTION"
        if "prohibited" in text_lower or "shall not" in text_lower or "must not" in text_lower:
            obligation = "PROHIBITION"
        elif "mandatory" in text_lower or "must" in text_lower or "shall" in text_lower:
            obligation = "MANDATORY"
            
        return {
            "substance": found_substance or "unknown",
            "property": found_property,
            "threshold": threshold_val,
            "unit": unit_val or "",
            "obligation": obligation,
            "raw_text": text
        }

    def extract_from_file(self, jsonl_path="data/processed/regulation_text.jsonl"):
        """Extracts all rules from jsonl file into structured DataFrame."""
        extracted = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                rule_id = record.get("rule_id", "")
                text = record.get("text", "")
                info = self.parse_text(text)
                info["rule_id"] = rule_id
                extracted.append(info)
        return pd.DataFrame(extracted)


def train_and_save(data_path="data/processed/regulation_text.jsonl", out="models/regulatory_ner.joblib"):
    extractor = RegulatoryEntityExtractor()
    df_rules = extractor.extract_from_file(data_path)
    Path(out).parent.mkdir(exist_ok=True)
    joblib.dump({"extracted_rules": df_rules}, out)
    return df_rules


def load_ner_model(path="models/regulatory_ner.joblib"):
    return joblib.load(path)


if __name__ == "__main__":
    rules_df = train_and_save()
    print(f"Extracted {len(rules_df)} regulatory constraint rules.")
