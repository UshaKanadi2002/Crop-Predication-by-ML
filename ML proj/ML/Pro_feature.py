# Step 2: Feature Selection & Preprocessing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the cleaned dataset
file_path = 'reduced_dataset.csv'
df = pd.read_csv(file_path)

# Select relevant features and target variable
features = ['area', 'season', 'crop_name', 'state_name', 'crop_type']
target = 'yield'
X = df[features]
y = df[target]

# One-hot encode categorical variables
categorical_cols = ['season', 'crop_name', 'state_name', 'crop_type']
ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
X_encoded = ohe.fit_transform(X[categorical_cols])

# Convert encoded features to DataFrame
encoded_df = pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out(categorical_cols))

# Combine encoded features with numerical columns
X_final = pd.concat([X[['area']].reset_index(drop=True), encoded_df], axis=1)

# Scale the numerical feature 'area'
scaler = StandardScaler()
X_final[['area']] = scaler.fit_transform(X_final[['area']])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# The dataset is now ready for model building.
