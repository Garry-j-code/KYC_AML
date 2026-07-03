import json
import random
from pathlib import Path

from faker import Faker

from src.config import KYC_DOCS_PATH

fake = Faker()


def make_doc(account_id: str, messy: bool = False) -> dict:
    doc = {
        "full_name": fake.name(),
        "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat(),
        "nationality": fake.country(),
        "address": fake.address().replace("\n", ", "),
        "id_number": fake.bothify("??######"),
        "account_id": account_id,
    }
    if messy:
        choice = random.choice(["drop", "date_fmt", "blank_name"])
        if choice == "drop":
            doc.pop(random.choice(["nationality", "id_number", "address"]))
        elif choice == "date_fmt":
            doc["date_of_birth"] = fake.date(pattern="%m-%d-%y")
        elif choice == "blank_name":
            doc["full_name"] = ""
    return doc


def doc_to_text(doc: dict) -> str:
    """Render a KYC doc dict as free-text for the intake agent."""
    lines = [f"{k.replace('_', ' ').title()}: {v}" for k, v in doc.items() if v]
    return "\n".join(lines)


def main(n: int = 200, messy_ratio: float = 0.2, out: Path | None = None) -> Path:
    out = out or KYC_DOCS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    docs = []
    for i in range(n):
        messy = random.random() < messy_ratio
        docs.append(make_doc(account_id=f"ACC{i:05d}", messy=messy))
    out.write_text(json.dumps(docs, indent=2))
    print(f"wrote {n} docs ({int(n * messy_ratio)} messy) -> {out}")
    return out


if __name__ == "__main__":
    main()
