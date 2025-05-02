import json
from pathlib import Path

files = [
    "faq_group1_cleaned.json", "faq_group2_cleaned.json", "faq_group3_cleaned.json", 
    "faq_group4_cleaned.json", "faq_manual_cleaned.json"
]
combined_data = []

for file in files:
    path = Path(f"data/processed/{file}")
    data = json.loads(path.read_text(encoding="utf-8"))
    combined_data.extend(data)

output_path = Path("data/processed/faq_combined.json")
output_path.write_text(json.dumps(combined_data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Đã gộp {len(combined_data)} mục vào {output_path}")