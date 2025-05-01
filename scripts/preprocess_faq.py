# preprocess_faq.py
import json, re, unicodedata, sys
from pathlib import Path
from jsonschema import validate, ValidationError

VALID_CATEGORIES = {"destination","weather","transport","scenery"}
SCHEMA_PATH = Path("data/schema_faq.jsonschema")

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", "", text)                 # strip HTML
    text = re.sub(r"[^\w\s.,!?-]", "", text)            # drop weird chars
    return " ".join(text.split()).strip()

def load_schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

def validate_and_clean(input_file="faq.json", output_file="faq_cleaned.json"):
    schema = load_schema()

    try:
        raw = json.loads(Path(input_file).read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"[ERROR] Cannot read {input_file}: {e}")

    cleaned = []
    for entry in raw:
        # 1. basic field check
        if not all(k in entry for k in ("question", "answer", "category")):
            print(f"❌ missing field → skip: {entry}")
            continue
        # 2. text normalisation
        entry["question"] = normalize_text(entry["question"])
        entry["answer"]   = normalize_text(entry["answer"])
        # 3. category quick check (faster fail than jsonschema)
        if entry["category"] not in VALID_CATEGORIES:
            print(f"❌ invalid category → skip: {entry}")
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
    validate_and_clean()
