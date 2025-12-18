from all_imports import *
from variables import *
from pipeline import *
from functions import *

# Modifying main df and creating test and train dfs
comp_engine03_df = create_rul_col(comp_engine03_df)
X_test_df = test_engine03_df.groupby('engine').last().drop(columns=not_unique_col(test_engine03_df)+['cycle', 'sensor10'])
y_test_df = rul_engine03_df['rul']
X_train_df = comp_engine03_df.drop(columns=not_unique_col(comp_engine03_df)+["engine", 'rul']+['cycle', 'sensor10'])
y_train_df = comp_engine03_df['rul']

# Clipping train data to focus more on the lower end of the data
y_train_clipped_df = y_train_df.clip(upper=120)# 112 or 120
rfr.fit(X_train_df, y_train_clipped_df)

# Scoring Model
rfr_model_score = rfr.score(X_test_df, y_test_df)
y_pred = rfr.predict(X_test_df)
rmse = np.sqrt(mean_squared_error(y_test_df, y_pred))

# Save the model in a pickle file
metadata = {
    "model_name": "RandomForestRegressor",
    "date_trained": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "features_used": list(X_train_df.columns),
    "r_squared_score" : rfr_model_score,
    "rmse" : rmse,
    "notes": "This model is using default values for RFR. A CV gridsearch did not improve metrics."
}

model_package = {
    "model": rfr,
    "metadata": metadata
}

with open("base_model.pkl", "wb") as f:
    pickle.dump(model_package, f)