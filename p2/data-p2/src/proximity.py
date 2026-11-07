
import pandas as pd
import numpy as np

asof_years = [2000, 2005, 2010, 2015]

for asof_year in asof_years:

    print(f"\nProcessing proximity for {asof_year}")

    # STEP 1: Load country-product matrix
    m = pd.read_parquet(f"matrix_A{asof_year}.parquet")
    M = m.to_numpy(dtype=np.int64)

    # STEP 2: Count countries exporting each product
    product_counts = M.sum(axis=0)

    # STEP 3: Count countries exporting both products
    cooccurrence = M.T @ M

    # STEP 4: Calculate denominator
    denominator = np.maximum(
        product_counts[:, None],
        product_counts[None, :]
    )

    # STEP 5: Calculate proximity
    proximity = np.divide(
        cooccurrence,
        denominator,
        out=np.zeros_like(cooccurrence, dtype=float),
        where=denominator != 0
    )

    # STEP 6: Convert to DataFrame
    proximity_df = pd.DataFrame(
    proximity,
    index=m.columns,
    columns=m.columns
    )

# STEP 7: Convert to long format
    output = (
    proximity_df
    .rename_axis(index="product_i", columns="product_j")
    .reset_index()
    .melt(
        id_vars="product_i",
        var_name="product_j",
        value_name="phi"
        )
    )

    # STEP 8: Save
    filename = f"proximity_A{asof_year}.parquet"
    output.to_parquet(filename, index=False)

    # STEP 9: Validation
    print("Matrix shape:", proximity.shape)
    print("Output shape:", output.shape)
    print("Minimum:", proximity.min())
    print("Maximum:", proximity.max())
    print("All values between 0 and 1:",
          np.all((proximity >= 0) & (proximity <= 1)))
    print("Symmetric:", np.allclose(proximity, proximity.T))
    print("Saved:", filename)

print("\nAll proximity files generated successfully!")
