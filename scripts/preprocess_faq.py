# preprocess_faq.py
import json, re, unicodedata, sys
from pathlib import Path
from jsonschema import validate, ValidationError
import argparse
VALID_CATEGORIES = {
    "destination", "weather", "transport", "scenery",
    "visa", "tour", "payment", "career", "company", "membership", "promotion",
    "destination_basics", "weather_best_time", "scenery_things_to_do", "visa_entry",
    "tour_booking", "pricing_payment_currency", "promotions_membership",
    "accommodation", "food_dining", "health_safety_insurance", "culture_etiquette",
    "events_festivals", "connectivity_sim_wifi", "budgeting_tips",
    "accessibility_family", "solo_female_safety", "sustainability_eco",
    "emergency_laws", "packing_checklist"
}
SCHEMA_PATH = Path("data/schema_faq.jsonschema")

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", "", text)                 # strip HTML
    text = re.sub(r"[^\w\s.,!?-]", "", text)            # drop weird chars
    return " ".join(text.split()).strip()

def load_schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

def validate_and_clean(input_file, output_file):
    schema = load_schema()

    try:
        raw = json.loads(Path(input_file).read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"[ERROR] Cannot read {input_file}: {e}")

    cleaned = []
    for entry in raw:
        # 1. basic field check
        if not all(k in entry for k in ("question", "answer")):
            print(f"❌ missing field → skip: {entry}")
            continue
        # Thay "category" bằng "tags" để đồng bộ với dataFromcallAPI.py
        if "category" in entry:
            entry["tags"] = [entry.pop("category")]
        if "tags" not in entry or not entry["tags"]:
            print(f"❌ missing tags → skip: {entry}")
            continue
        # 2. text normalisation
        entry["question"] = normalize_text(entry["question"])
        entry["answer"] = normalize_text(entry["answer"])
        # 3. tags quick check
        if not all(tag in VALID_CATEGORIES for tag in entry["tags"]):
            print(f"❌ invalid tags → skip: {entry}")
            continue
        # 4. JSON-Schema validation
        try:
            validate(entry, schema)
            cleaned.append(entry)
        except ValidationError as ve:
            print(f"❌ schema violation → skip: {ve.message}")
            continue

    Path(output_file).write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅  Saved {len(cleaned)} valid entries → {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess FAQ data.")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    validate_and_clean(args.input, args.output)