# Step 3: Model Building - Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the cleaned dataset
file_path = 'reduced_dataset.csv'
df = pd.read_csv(file_path)

# Select relevant features and target variable
features = ['area', 'season', 'crop_name', 'state_name', 'crop_type']
target = 'yield'
X = df[features]
y = df[target]

# One-hot encode categorical variables
ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
X_encoded = ohe.fit_transform(X[['season', 'crop_name', 'state_name', 'crop_type']])
encoded_df = pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out())

# Combine encoded features with numerical columns
X_final = pd.concat([X[['area']].reset_index(drop=True), encoded_df], axis=1)

# Scale the numerical feature 'area'
scaler = StandardScaler()
X_final[['area']] = scaler.fit_transform(X_final[['area']])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
