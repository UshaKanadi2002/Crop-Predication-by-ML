# Step 4: Model Comparison & Cross-Validation

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the cleaned dataset
file_path = 'reduced_dataset.csv'
df = pd.read_csv(file_path)

df.fillna(0, inplace=True)

# ----------------------------
# Preprocessing
# ----------------------------

season_map = {'Kharif': 1, 'Rabi': 2, 'Summer': 3, 'Winter': 4, 'Whole Year': 5}
df['season_encoded'] = df['season'].map(season_map).fillna(0)

features = ['area', 'production', 'season_encoded', 'state_name', 'crop_type']
target = 'yield'
X = df[features]
y = df[target]

ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
X_encoded = ohe.fit_transform(X[['state_name', 'crop_type']])
encoded_df = pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out())

X_final = pd.concat([X[['area', 'production', 'season_encoded']].reset_index(drop=True),
                     encoded_df.reset_index(drop=True)], axis=1)

scaler = StandardScaler()
X_final[['area', 'production', 'season_encoded']] = scaler.fit_transform(
    X_final[['area', 'production', 'season_encoded']]
)

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# ----------------------------
# Model Evaluation Function
# ----------------------------

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, mae, rmse

# ----------------------------
# Model Comparison
# ----------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=100)
}

print("--- Model Comparison ---")

for name, model in models.items():
    r2, mae, rmse = evaluate_model(model, X_train, y_train, X_test, y_test)
    print(f"{name} Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}\n")

# ----------------------------
# Cross-Validation
# ----------------------------

print("--- Cross-Validation Scores (R²) ---")

for name, model in models.items():
    cv_scores = cross_val_score(model, X_final, y, cv=5, scoring='r2')
    print(f"{name}: Mean R² = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}")
