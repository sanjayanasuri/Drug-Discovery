import pandas as pd
from chembl_webresource_client.new_client import new_client

print("Starting ChEMBL download...")

target_id = "CHEMBL243"  # HIV-1 protease
activity = new_client.activity

# Filter down the space + only fetch a limited number of records
qs = activity.filter(
    target_chembl_id=target_id,
    standard_type="IC50",
    standard_units="nM"
).only(
    ['canonical_smiles', 'standard_value', 'standard_flag']
)

# Limit to first N results so it doesn't hang forever
N = 2000  # you can adjust this up/down
res = qs[:N]

df = pd.DataFrame(res)

# Keep only what we need
cols = ["canonical_smiles", "standard_value", "standard_flag"]
df = df[cols].dropna(subset=["canonical_smiles", "standard_value"])

# Rename to match main.py
df = df.rename(columns={
    "canonical_smiles": "smiles",
    "standard_value": "activity_value"
})

df = df.drop_duplicates(subset=["smiles", "activity_value"])

df.to_csv("bioactivity.csv", index=False)
print("Bioactivity data saved to bioactivity.csv")
print("Done. Rows downloaded:", len(df))
print(df.head())
