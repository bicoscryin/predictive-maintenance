def create_failure_col(multi_engine_df):
    loop_df = multi_engine_df.copy()
    loop_df["failure"] = 0
    for i in range(1,101):
        engine_loop_df = loop_df.loc[loop_df['engine'] == i]
        engine_loop_df.loc[engine_loop_df['cycle'] == len(engine_loop_df), "failure"] = 1
        loop_df.loc[loop_df['engine'] == i] = engine_loop_df
    return loop_df
def create_rul_col(multi_engine_df):
    loop_df = multi_engine_df.copy()
    loop_df["rul"] = 0
    for i in range(1,101):
        loop_df.loc[loop_df['engine'] == i, 'rul'] = loop_df.loc[loop_df['engine'] == i, 'cycle'].max() - loop_df.loc[loop_df['engine'] == i, 'cycle']
    return loop_df
def not_unique_col(loop_df):
    loop_list = []
    loop_df = loop_df.copy()
    for col in loop_df.columns:
        if (loop_df[col].to_numpy()[0] == loop_df[col].to_numpy()).all(0):
            loop_list.append(col)
    return loop_list