from all_imports import *
from variables import *

# Model Code
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),   # Fill missing with mean
    ("scaler", MinMaxScaler())                     # Scale to [0, 1]
])

categorical_transformer = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore")) 
])

# ColumnTransformer applies the correct transformer to each column set
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


rfr = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

lr = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

dtr = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

xgbr = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(random_state=42))
])

ridge_reg = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(random_state=42))
])

lasso_reg = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Lasso(random_state=42))
])