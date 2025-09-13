import pandas as pd
import json

# Load the CSV file
df = pd.read_csv("label_names.csv")  # Make sure this path is correct

# Create a dictionary: ClassId -> SignName
label_map = {int(row["ClassId"]): row["SignName"] for idx, row in df.iterrows()}

# Save as JSON
with open("label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

print("✅ Corrected label_map.json created!")
