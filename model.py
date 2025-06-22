import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
import os


df = pd.read_csv('honda_toyota_ca.csv')

# Train/test split
X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=df[['make', 'model']], test_size=0.2, shuffle=True, random_state=42)

# Preprocessing pipeline
cat_index = [2,3,5]

cat_features_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder()),
        ("selector", SelectPercentile(mutual_info_regression, percentile=50))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_features_transformer, cat_index)
    ]
)

# Modeling pipeline

from sklearn.ensemble import GradientBoostingRegressor

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(random_state=42))
    ]
)

# Fit the model

model.fit(X_train, y_train)

# Score the model (R2)

print(model.score(X_test, y_test))

# Save the model
os.makedirs('model', exist_ok=True)
dump(model, 'model/model.joblib')