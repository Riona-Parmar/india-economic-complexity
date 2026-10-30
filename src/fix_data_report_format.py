
from pathlib import Path

path = Path("outputs/reports/data_report.md")
text = path.read_text(encoding="utf-8")

replacements = {
    "calculatedRCA3": "calculated RCA3",
    "remainbelow": "remain below",
    "outcomesand": "outcomes and",
}

for old, new in replacements.items():
    text = text.replace(old, new)

# Add concrete evidence to the existing RCA caveat.
old_caveat = (
    "- The Atlas source RCA differs from the independently calculated "
    "Balassa RCA for some records; this remains an audit item."
)

new_caveat = (
    "- The Atlas source `export_rca` differs from the independently "
    "calculated standard Balassa RCA for some records; the cause remains "
    "unresolved. For Mali (MLI), HS4 product 5203, in 2022, source RCA "
    "was 15,895.525391 while calculated RCA was 2,308.718728. The "
    "calculated result agrees with the RCA implied by the source "
    "`global_market_share` (approximately 2,308.717252). Keep the "
    "calculated RCA as the project metric and retain the source value "
    "for audit; do not claim the cause is established."
)

text = text.replace(old_caveat, new_caveat)

path.write_text(text, encoding="utf-8")

print(f"Updated {path}")
print("\nRemaining formatting issues:")
for phrase in ["calculatedRCA3", "remainbelow", "outcomesand"]:
    print(f"{phrase}: {phrase in text}")

print("\nRCA audit note present:", "15,895.525391" in text)
