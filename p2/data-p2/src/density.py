
import pandas as pd
import numpy as np

asof_years = [2000, 2005, 2010, 2015]

for asof_year in asof_years:

    print(f"\nProcessing density for {asof_year}")

    # STEP 1: Load country-product matrix
    m = pd.read_parquet(f"matrix_A{asof_year}.parquet")
    M = m.to_numpy(dtype=np.float64)

    # STEP 2: Load proximity data
    proximity_df = pd.read_parquet(
        f"proximity_A{asof_year}.parquet"
    )

    # Convert long format into square matrix
    phi_df = proximity_df.pivot(
        index="product_i",
        columns="product_j",
        values="phi"
    )

    # Match product ordering
    phi_df = phi_df.reindex(
        index=m.columns,
        columns=m.columns
    )

    Phi = phi_df.to_numpy(dtype=np.float64)

    # STEP 3: Calculate numerator
    numerator = M @ Phi.T

    # STEP 4: Calculate denominator
    denominator = Phi.sum(axis=1)

    # STEP 5: Calculate density
    density = np.divide(
        numerator,
        denominator[None, :],
        out=np.zeros_like(numerator),
        where=denominator[None, :] != 0
    )

    # STEP 6: Convert to DataFrame
    density_df = pd.DataFrame(
        density,
        index=m.index,
        columns=m.columns
    )

    density_df.index.name = "country"
    density_df.columns.name = "product"

    # STEP 7: Convert to long format
    output = density_df.reset_index().melt(
        id_vars="country",
        var_name="product",
        value_name="density"
    )

    output.insert(0, "asof_year", asof_year)

    # STEP 8: Save
    filename = f"density_A{asof_year}.parquet"
    output.to_parquet(filename, index=False)

    # STEP 9: Validate
    print("Density matrix shape:", density.shape)
    print("Output shape:", output.shape)
    print("Minimum density:", output["density"].min())
    print("Maximum density:", output["density"].max())
    print(
        "All values between 0 and 1:",
        output["density"].between(0, 1).all()
    )
    print("Missing values:", output["density"].isna().sum())
    print("Saved:", filename)

print("\nAll density files generated successfully!")
