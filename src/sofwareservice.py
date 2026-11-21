
import requests
import pandas as pd
import time

def get_data(country, indicator):
    url = (
        f"https://api.worldbank.org/v2/country/{country}/"
        f"indicator/{indicator}"
    )

    for attempt in range(5):
        try:
            response = requests.get(
                url,
                params={
                    "format": "json",
                    "per_page": 100,
                    "date": "2000:2025"
                },
                timeout=30
            )

            response.raise_for_status()
            data = response.json()

            if not isinstance(data, list) or len(data) < 2:
                raise ValueError(f"Unexpected API response: {data}")

            records = data[1]

            return pd.DataFrame([
                {
                    "year": int(r["date"]),
                    country: r["value"]
                }
                for r in records
                if r["value"] is not None
            ])

        except Exception as e:
            print(f"Attempt {attempt + 1}/5 failed: {e}")

            if attempt == 4:
                raise

            time.sleep(3 * (attempt + 1))


# Download all four datasets
print("Downloading World Bank data...")

india_computer = get_data("IND", "BX.GSR.CCIS.CD")
world_computer = get_data("WLD", "BX.GSR.CCIS.CD")
india_services = get_data("IND", "BX.GSR.NFSV.CD")
world_services = get_data("WLD", "BX.GSR.NFSV.CD")

# Merge datasets
df = india_computer.merge(world_computer, on="year")
df = df.merge(india_services, on="year")
df = df.merge(world_services, on="year")

# Calculate RCA
df["RCA"] = (
    (df["IND_x"] / df["IND_y"]) /
    (df["WLD_x"] / df["WLD_y"])
)

df = df.sort_values("year")

# Save results
df.to_csv("india_computer_services_rca.csv", index=False)

print("\nRCA DATA:")
print(df[["year", "RCA"]].to_string(index=False))

print("\nCSV saved successfully!")
