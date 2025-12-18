from all_imports import *
from variables import *
from pipeline import *
from functions import *

with open("base_model.pkl", "rb") as f:
    loaded_package= pickle.load(f)
model = loaded_package["model"]
metadata = loaded_package['metadata']

comp_engine03_df = create_rul_col(comp_engine03_df)
X_test_df = test_engine03_df.groupby('engine').last().drop(columns=not_unique_col(test_engine03_df)+['cycle', 'sensor10'])
y_test_df = rul_engine03_df['rul']
X_train_df = comp_engine03_df.drop(columns=not_unique_col(comp_engine03_df)+["engine", 'rul']+['cycle', 'sensor10'])
y_train_df = comp_engine03_df['rul']
y_train_clipped_df = y_train_df.clip(upper=120)# 112 or 120 or 108
# rfr.fit(X_train_df, y_train_clipped_df)
rfr_model = model.named_steps["regressor"]
num_fet_name_list = [f"{sensor_names[fet]}" for fet in numeric_features]
user_defined_df = pd.DataFrame(columns=X_train_df.columns)
user_defined_rul_list = []
pred_rul_df = pd.DataFrame(columns=['engine', 'rul', 'label'])

# Theme


st.title("Predictive Maintenance on a Jet Engine")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data", "Indicators of Engine Health", "Test Engine Comparison", "Predicting Engine Failure", "Categorizing Potential Failures"])

# Overall Data
with tab1:
    with st.container():
        st.header("Model Data")
        st.write("The graph shows how long the engine ran before failure. This is measuered in cycles. One cycle is from initial start to when the engine is shut down.")
        comp03_list = []
        for idx in range(1, 101):
            comp03_list.append(comp_engine03_df.loc[comp_engine03_df['engine'] == idx,'cycle'].max())
        fig, ax = plt.subplots(figsize=(2,5))
        plt.hist(comp03_list, bins=20)
        # plt.title('Average Max Engine Life')
        plt.ylabel('Frequency')
        plt.xlabel('Max Engine Cycles')
        st.plotly_chart(fig)

with tab2:
    with st.container():
        # Engine health endicators
        st.header("Indicators of Engine Health")
        st.write("There are 21 total sensors in the data. The 5 most significant are depicted below.")
        importances = pd.DataFrame(rfr_model.feature_importances_, columns=["Feature Importance"], index=num_fet_name_list).sort_values(by="Feature Importance").tail()
        fig, ax = plt.subplots(figsize=(12,4))
        plt.bar(importances.index, importances["Feature Importance"])
        plt.xticks(rotation=25)
        st.pyplot(fig)


fig, ax = plt.subplots(figsize=(2,5))

# Overview of how jet engines work
with st.container():
    st.header("About Jet Engines")
    st.write("Jet engined take in air, compress it, mix it with fuel, ignite it, and use the resulting combustion for thrust.  " \
    "The three main sections of the engine are the compressors (LPC and HPC), the turbines (HPT and LPT), and the combuster." \
    " Low pressure (LPC and LPT) areas are connected by the N2 shaft. High pressure (HPC and HPT) areas are connected by the N1 shaft.")
    st.image("../img/turbofan_model.png")

# compare different engines and different sensors
with tab3:
    with st.container():
        st.header("Test Engine Comparison")
        st.write("Each Engine was ran till failure. The below graph compares the selected sensor data between two test engines.")
        cols = st.columns(2)
        with cols[0]:
            engine_1 = st.selectbox("Blue", range(1,101), 15) #16
        with cols[1]:
            engine_2 = st.selectbox("Orange", range(1,101), 27)
        col = st.selectbox("Select the Sensor", num_fet_name_list, 7)
        col = list(sensor_names.keys())[list(sensor_names.values()).index(col)]
        engine_series = comp_engine03_df.loc[comp_engine03_df['engine'] == engine_1, col]
        cycle_series = comp_engine03_df.loc[comp_engine03_df['engine'] == engine_1, 'cycle']
        plt.scatter(cycle_series, engine_series, label=f"Engine {engine_1}")
        # fig = px.scatter(cycle_series, engine_series, trendline="ols", trendline_options=dict(log_x=True))
        engine_series_2 = comp_engine03_df.loc[comp_engine03_df['engine'] == engine_2, col]
        cycle_series_2 = comp_engine03_df.loc[comp_engine03_df['engine'] == engine_2, 'cycle']
        plt.scatter(cycle_series_2, engine_series_2, label=f"Engine {engine_2}")
        # plt.title(f"comp_engine03_df")
        plt.xlabel("Cycles")
        plt.ylabel(f"{sensor_names[col]} ({col})")
        # plt.gca().invert_xaxis()
        # plt.legend()
        st.plotly_chart(fig)

with tab4:
    # Predicting using custom variables or presets
    with st.container():
        # with tab2:
        st.header("Predicting Engine Failure")
        st.markdown("Raw data used to predict Remaining Useful Life.  \n"\
                    "(The sliders are set to a cycle from the data by default)  \n" \
        "You can change these values by selecting manual.")
        cols = st.columns(2)
        with cols[0]:
            what_level = st.selectbox("Input method", ['Preset', 'Manual'])
        if what_level == "Preset":
            with cols[1]:
                example_engine = st.selectbox("Example Engine", range(1,101), 15)
            # st.dataframe(X_test_df.iloc[[example_engine]], column_config=sensor_names)
            rul = model.predict(X_test_df.loc[[example_engine]])
        if what_level == 'Manual':
            st.sidebar.header("Sensor Values")
            example_engine = 16
            sensor2 = st.sidebar.slider(f"{sensor_names['sensor2']}", 640.0, 650.0, X_test_df.loc[example_engine,"sensor2"])
            sensor3 = st.sidebar.slider(f"{sensor_names['sensor3']}", 1550.0, 1620.0, X_test_df.loc[example_engine,"sensor3"])
            sensor4 = st.sidebar.slider(f"{sensor_names['sensor4']}", 1370.0, 1450.0, X_test_df.loc[example_engine,"sensor4"])
            sensor6 = st.sidebar.slider(f"{sensor_names['sensor6']}", 21.3, 21.8, X_test_df.loc[example_engine,"sensor6"])
            sensor7 = st.sidebar.slider(f"{sensor_names['sensor7']}", 540.0, 580.0, X_test_df.loc[example_engine,"sensor7"])
            sensor8 = st.sidebar.slider(f"{sensor_names['sensor8']}", 2386.0, 2389.0, X_test_df.loc[example_engine,"sensor8"])
            sensor9 = st.sidebar.slider(f"{sensor_names['sensor9']}", 9000.0, 9300.0, X_test_df.loc[example_engine,"sensor9"])
            sensor11 = st.sidebar.slider(f"{sensor_names['sensor11']}", 46.0, 48.5, X_test_df.loc[example_engine,"sensor11"])
            sensor12 = st.sidebar.slider(f"{sensor_names['sensor12']}", 517.0, 538.0, X_test_df.loc[example_engine,"sensor12"])
            sensor13 = st.sidebar.slider(f"{sensor_names['sensor13']}", 2386.0, 2389.0, X_test_df.loc[example_engine,"sensor13"])
            sensor14 = st.sidebar.slider(f"{sensor_names['sensor14']}", 8050.0, 8300.0, X_test_df.loc[example_engine,"sensor14"])
            sensor15 = st.sidebar.slider(f"{sensor_names['sensor15']}", 8.1, 8.6, X_test_df.loc[example_engine,"sensor15"])
            sensor17 = st.sidebar.slider(f"{sensor_names['sensor17']}", 387, 400, X_test_df.loc[example_engine,"sensor17"])
            sensor20 = st.sidebar.slider(f"{sensor_names['sensor20']}", 38.0, 40.0, X_test_df.loc[example_engine,"sensor20"])
            sensor21 = st.sidebar.slider(f"{sensor_names['sensor21']}", 22.5, 24.0, X_test_df.loc[example_engine,"sensor21"])
            slider_df = pd.DataFrame([[sensor2, sensor3, sensor4, sensor6, sensor7, sensor8,
            sensor9, sensor11, sensor12, sensor13, sensor14,
            sensor15, sensor17, sensor20, sensor21]], columns=X_train_df.columns)
            rul = model.predict(slider_df)
        with cols[0]:
            st.metric(
                "**Remaining Useful Life** (cycles)",
                math.ceil(rul[0]),
                help="This is the amount of cycles until the engine is predicted to fail.",
                border=True
            )
        # st.write(f"**Predicted Remaining Useful Life: {math.ceil(rul[0])} cycles**")

with tab5:
    # Sectioning off groups for maintenance schedules
    with st.container():
        st.header("Categorizing Potential Failures")
        st.markdown("I classified engine health based on the remaining useful life. All engines in the \"Good\" category will not need maintenance in the near future."\
                    " All engines in the \"Warning\" category will need mainenance scheduled. All engines in the \"Danger\" category will need maintenance performed as soon as possible." \
                    " The dotted line marks the Danger zone. Since I did not want to over predict how long an engine would last, I selected 40 cycles. This was "
                    "the closest to the margine of error for the predictions.")
        for i in range(1,101):
            pred_loop = math.ceil(list(model.predict(X_test_df.loc[[i]]))[0])
            if pred_loop > 80:
                pred_rul_df.loc[i, 'engine'] = int(i)
                pred_rul_df.loc[i, 'rul'] = pred_loop
                pred_rul_df.loc[i, 'label'] = 'Good'
            elif pred_loop > 40:
                pred_rul_df.loc[i, 'engine'] = int(i)
                pred_rul_df.loc[i, 'rul'] = pred_loop
                pred_rul_df.loc[i, 'label'] = 'Warning'
            else:
                pred_rul_df.loc[i, 'engine'] = int(i)
                pred_rul_df.loc[i, 'rul'] = pred_loop
                pred_rul_df.loc[i, 'label'] = 'Danger'
        # with cols[1]:
        fig = px.scatter(pred_rul_df, y='rul', x='engine', color='label', hover_data=['engine'], color_discrete_sequence=[ 'orange', "blue", "red"])
        fig.add_shape(type="line",
                x0=-5, 
                y0=40, 
                x1=120, 
                y1=40,
                line=dict(
                color="tomato",
                width=4,
                dash="dot",
        ))
        st.plotly_chart(fig)

# with st.container():
#     st.header("Prediction Accuracy")
#     fig, ax = plt.subplots(figsize=(2,5))
#     pred_list = []
#     for i in range(1,101):
#         pred_list.append(math.ceil(list(model.predict(X_test_df.loc[[i]]))[0]))
#     error_pred_df = pd.DataFrame()
#     error_pred_df['Prediction RUL'] = pred_list
#     error_pred_df['Real RUL'] = y_test_df
#     # plt.scatter(pred_list, y_test_df)
#     fig = px.scatter(error_pred_df, x='Prediction RUL', y='Real RUL', trendline="ols")
#     fig['layout']['xaxis']['autorange'] = "reversed"
#     st.plotly_chart(fig)