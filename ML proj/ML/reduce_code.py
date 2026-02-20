import pandas as pd
# Load the dataset
file_path = "your_dataset.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Drop the specified columns
columns_to_remove = ["state_code", "district_code", "area_unit", "yield_unit", "production_unit"]
df_reduced = df.drop(columns=columns_to_remove, errors="ignore")

# Save the cleaned dataset
df_reduced.to_csv("reduced_dataset.csv", index=False)

print("Dataset cleaned and saved as 'reduced_dataset.csv'.")
