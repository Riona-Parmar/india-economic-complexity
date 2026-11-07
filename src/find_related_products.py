
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

predictions = pd.read_csv(
    ROOT / "outputs" / "india_shap_explanations.csv",
    dtype={"product": str}
)

trade = pd.read_parquet(
    ROOT / "data" / "processed" / "trade_rca_v1.parquet"
)

all_related = []

for year in [2010, 2015]:

    print(f"\nFinding related products for {year}")

    phi = pd.read_parquet(
        ROOT / "p2" / "data-p2" /
        "proximity" / f"proximity_A{year}.parquet"
    )

    # India's latest competitive products
    latest = (
        trade[
            (trade["country"] == "IND") &
            (trade["year"] <= year)
        ]
        .sort_values("year")
        .groupby("product")
        .tail(1)
    )

    competitive_products = set(
        latest.loc[latest["m"] == 1, "product"]
    )

    top_candidates = predictions[
        predictions["asof_year"] == year
    ].sort_values(
        "probability", ascending=False
    ).head(10)

    for _, candidate in top_candidates.iterrows():

        code = candidate["product"]

        # Check both directions
        related_1 = phi[
            (phi["product_i"] == code)
        ][["product_j", "phi"]].rename(
            columns={"product_j": "related_product"}
        )

        related_2 = phi[
            (phi["product_j"] == code)
        ][["product_i", "phi"]].rename(
            columns={"product_i": "related_product"}
        )

        related = pd.concat(
            [related_1, related_2],
            ignore_index=True
        )

        related = related[
            related["related_product"] != code
        ]

        # Keep products India already exports competitively
        related = related[
            related["related_product"].isin(
                competitive_products
            )
        ]

        related = (
            related.sort_values("phi", ascending=False)
            .drop_duplicates("related_product")
            .head(5)
        )

        for _, item in related.iterrows():

            all_related.append({
                "asof_year": year,
                "candidate_product": code,
                "candidate_probability": candidate["probability"],
                "related_product": item["related_product"],
                "proximity": item["phi"]
            })

related_df = pd.DataFrame(all_related)

output_path = (
    ROOT / "outputs" /
    "india_nearest_related_products.csv"
)

related_df.to_csv(output_path, index=False)

print("\nSaved:", output_path)
print("\nRESULTS:")
print(related_df.to_string(index=False))
