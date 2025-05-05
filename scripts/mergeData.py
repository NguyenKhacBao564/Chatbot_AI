# import json
# from pathlib import Path

# # files = [
# #     "faq_group1_cleaned.json", "faq_group2_cleaned.json", "faq_group3_cleaned.json", 
# #     "faq_group4_cleaned.json", "faq_manual_cleaned.json"
# # ]
# files = [
#     "faq_north.json", "faq_central.json", "faq_south.json"
# ]
# combined_data = []

# for file in files:
#     # path = Path(f"data/processed/{file}")
#     path = Path(f"data/raw/{file}")
#     data = json.loads(path.read_text(encoding="utf-8"))
#     combined_data.extend(data)

# # output_path = Path("data/processed/faq_combined.json")
# output_path = Path("data/raw/faq_combined_ncs.json")
# output_path.write_text(json.dumps(combined_data, ensure_ascii=False, indent=2), encoding="utf-8")
# print(f"Đã gộp {len(combined_data)} mục vào {output_path}")

import json
from pathlib import Path

files = ["faqupdate_north.json", "faqupdate_central.json", "faqupdate_south.json", "faq_general.json"]
combined_data = []
current_id = 0

for file in files:
    path = Path(f"data/raw/{file}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        item["id"] = current_id
        if "location" not in item:
            item["location"] = "General"
        combined_data.append(item)
        current_id += 1

output_path = Path("data/raw/faq_combined.json")
output_path.write_text(json.dumps(combined_data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Đã gộp {len(combined_data)} mục vào {output_path}")