# Step 1: Exploratory Data Analysis (EDA)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = 'reduced_dataset.csv'
df = pd.read_csv(file_path)

# Basic dataset info
print("Dataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nSummary Statistics:\n", df.describe())

# Check for unique values in categorical columns
categorical_cols = ['year', 'state_name', 'district_name', 'season', 'crop_name', 'crop_type']
for col in categorical_cols:
    print(f"\nUnique values in {col}: {df[col].nunique()}")

# Correlation heatmap for numerical features
plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Distribution of yield
plt.figure(figsize=(6, 4))
sns.histplot(df['yield'], kde=True)
plt.title("Yield Distribution")
plt.xlabel("Yield (Tonnes/Hectare)")
plt.show()

# Scatter plot: Area vs Yield
plt.figure(figsize=(6, 4))
sns.scatterplot(x='area', y='yield', data=df)
plt.title("Area vs Yield")
plt.xlabel("Area (Hectare)")
plt.ylabel("Yield (Tonnes/Hectare)")
plt.show()

# Boxplot: Yield by Season
plt.figure(figsize=(8, 4))
sns.boxplot(x='season', y='yield', data=df)
plt.title("Yield by Season")
plt.xlabel("Season")
plt.ylabel("Yield (Tonnes/Hectare)")
plt.xticks(rotation=45)
plt.show()
