"""One-off cleanup: bring prediction files in line with the P5 dashboard contract.

Contract: outputs/predictions_*.parquet with columns
    asof_year, country, product, model, score
and model in {rca_persistence, proximity, xgboost}.

Safe to re-run: moves are skipped if already done, and `model` is overwritten
with the same value.
"""

from pathlib import Path
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
P2_PRED = ROOT / "p2" / "data-p2" / "predictions"

COLUMNS = ["asof_year", "country", "product", "model", "score"]

# file in outputs/ -> (model name, original P2 location or None)
FILES = {
    "predictions_xgboost_v1.parquet": ("xgboost", None),
    "predictions_rca.parquet": ("rca_persistence", P2_PRED / "predictions_rca.parquet"),
    "predictions_density.parquet": ("proximity", P2_PRED / "predictions_density.parquet"),
}


def main() -> None:
    for name, (model, source) in FILES.items():
        dest = OUTPUTS / name

        if source is not None and source.exists():
            if dest.exists():
                raise FileExistsError(f"Refusing to overwrite existing {dest}")
            shutil.move(str(source), str(dest))
            print(f"Moved {source.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")

        df = pd.read_parquet(dest)
        df["model"] = model
        df = df[COLUMNS]
        df.to_parquet(dest, index=False)
        print(f"Saved {dest.relative_to(ROOT)}: {len(df):,} rows, model={model!r}")


if __name__ == "__main__":
    main()
