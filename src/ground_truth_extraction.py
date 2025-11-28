# Utility that extracts the ground-truth vulnerabilities from "smartbugs-curated/vulnerabilities.json".
import json
from vulnerabilities_constants import KEYS_TO_CATEGORIES, CATEGORIES

def extract_ground_truth(json_path="smartbugs-curated/vulnerabilities.json"):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    gt = {}
    for item in data:
        file_name = item["name"]
        vulns = item.get("vulnerabilities", [])
        # Initialize all categories to 0.
        cat_map = {k: 0 for k in CATEGORIES}
        # Mark vulnerable categories as 1.
        for v in vulns:
            raw = v["category"].lower().strip()
            if raw in KEYS_TO_CATEGORIES:
                key = KEYS_TO_CATEGORIES[raw]
                cat_map[key] = 1
        gt[file_name] = cat_map
    return gt