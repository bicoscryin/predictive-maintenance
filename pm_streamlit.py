from all_imports import *
from variables import *
from pipeline import *
from functions import *

with open("base_model.pkl", "rb") as f:
    loaded_package= pickle.load(f)
model = loaded_package["model"]
metadata = loaded_package['metadata']

comp_engine03_df = create_rul_col(comp_engine03_df)
X_test_df = test_engine03_df.groupby('engine').last().drop(columns=not_unique_col(test_engine03_df)+['cycle'])
y_test_df = rul_engine03_df['rul']
X_train_df = comp_engine03_df.drop(columns=not_unique_col(comp_engine03_df)+["engine", 'rul']+['cycle'])
y_train_df = comp_engine03_df['rul']
y_train_clipped_df = y_train_df.clip(upper=120)# 112 or 120 or 108
# rfr.fit(X_train_df, y_train_clipped_df)
rfr_model = model.named_steps["regressor"]
num_fet_name_list = [f"{sensor_names[fet]} ({fet})" for fet in numeric_features]


st.title("Predictive Maintenance on a Jet Engine")

tab1, tab2 = st.tabs(["Data", "Prediction"])
with tab1:
    st.header("Model Data")
    st.write("How long the engine ran before failure.")
    comp03_list = []
    for idx in range(1, 101):
        comp03_list.append(comp_engine03_df.loc[comp_engine03_df['engine'] == idx,'cycle'].max())
    fig, ax = plt.subplots(figsize=(12,4))
    plt.hist(comp03_list, bins=20)
    plt.title('Distribution of Max Cycles')
    plt.ylabel('Frequency')
    plt.xlabel('Max Engine Cycles')
    st.plotly_chart(fig)

    importances = pd.DataFrame(rfr_model.feature_importances_, columns=["Feature Importance"], index=num_fet_name_list).sort_values(by="Feature Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(12,4))
    plt.bar(importances.index, importances["Feature Importance"])
    plt.xticks(rotation=75)
    st.plotly_chart(fig)


    fig, ax = plt.subplots(figsize=(12,4))
    def plt_scatter(x, y, col, title, x_label, y_label, invertx = True):
        engine_series = x[col]
        cycle_series = y
        fig, ax = plt.subplots(figsize=(12,4))
        plt.scatter(cycle_series, engine_series, label="comp_engine03_df")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if invertx == True:
            plt.gca().invert_xaxis()
        plt.legend()
        st.plotly_chart(fig)
    plt_scatter(X_train_df, y_train_df, 'sensor11', f"Engine {1}", "Remaining Usefull Life", f"{sensor_names['sensor11']} ({'sensor11'})", invertx = True)
    plt_scatter(X_train_df, y_train_df, 'sensor14', f"Engine {1}", "Remaining Usefull Life", f"{sensor_names['sensor14']} ({'sensor14'})", invertx = True)
with tab2:
    st.header("Model Data")
    st.write("Raw data used to predict Remaining Usefull Life.")
    sensor2 = st.slider(f"{sensor_names['sensor2']}", 640.0, 650.0, X_test_df.loc[1,"sensor2"])
    sensor3 = st.slider(f"{sensor_names['sensor3']}", 1550.0, 1620.0, X_test_df.loc[1,"sensor3"])
    sensor4 = st.slider(f"{sensor_names['sensor4']}", 1370.0, 1450.0, X_test_df.loc[1,"sensor4"])
    sensor6 = st.slider(f"{sensor_names['sensor6']}", 21.3, 21.8, X_test_df.loc[1,"sensor6"])
    sensor7 = st.slider(f"{sensor_names['sensor7']}", 540.0, 580.0, X_test_df.loc[1,"sensor7"])
    sensor8 = st.slider(f"{sensor_names['sensor8']}", 2386.0, 2389.0, X_test_df.loc[1,"sensor8"])
    sensor9 = st.slider(f"{sensor_names['sensor9']}", 9000.0, 9300.0, X_test_df.loc[1,"sensor9"])
    sensor10 = st.slider(f"{sensor_names['sensor10']}", 1.2, 1.4, X_test_df.loc[1,"sensor10"])
    sensor11 = st.slider(f"{sensor_names['sensor11']}", 46.0, 48.5, X_test_df.loc[1,"sensor11"])
    sensor12 = st.slider(f"{sensor_names['sensor12']}", 517.0, 538.0, X_test_df.loc[1,"sensor12"])
    sensor13 = st.slider(f"{sensor_names['sensor13']}", 2386.0, 2389.0, X_test_df.loc[1,"sensor13"])
    sensor14 = st.slider(f"{sensor_names['sensor14']}", 8050.0, 8300.0, X_test_df.loc[1,"sensor14"])
    sensor15 = st.slider(f"{sensor_names['sensor15']}", 8.1, 8.6, X_test_df.loc[1,"sensor15"])
    sensor17 = st.slider(f"{sensor_names['sensor17']}", 387, 400, X_test_df.loc[1,"sensor17"])
    sensor20 = st.slider(f"{sensor_names['sensor20']}", 38.0, 40.0, X_test_df.loc[1,"sensor20"])
    sensor21 = st.slider(f"{sensor_names['sensor21']}", 22.5, 24.0, X_test_df.loc[1,"sensor21"])
    slider_df = pd.DataFrame([[sensor2, sensor3, sensor4, sensor6, sensor7, sensor8,
       sensor9, sensor10, sensor11, sensor12, sensor13, sensor14,
       sensor15, sensor17, sensor20, sensor21]], columns=X_train_df.columns)
    rul = model.predict(slider_df)
    st.write(f"Predicted Remaining Usefull Life: {rul[0]} cycles")