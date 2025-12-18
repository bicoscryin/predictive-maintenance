import pandas as pd
import numpy as np
import os
from functions import *

data_directory = "\\".join(f"{os.getcwd()}".split("\\")[0:len(f"{os.getcwd()}".split("\\"))-1] + ["data"])

# Engine column names
index_names = {
"engine" : "Engine No.",
"cycle" : "Time, In Cycles"
}

setting_names = {
"setting1" : "Operation Altitude (K ft.)",
"setting2" : "Mach Number",
"setting3" : "TRA (degrees)"
}

sensor_names = {
"sensor1" : "Fan Inlet Temperature (◦R)",
"sensor2" : "LPC Outlet Temperature (◦R)",
"sensor3" : "HPC Outlet Temperature (◦R)",
"sensor4" : "LPT Outlet Temperature (◦R)",
"sensor5" : "Fan Inlet Pressure (psia)",
"sensor6" : "Bypass-Duct Pressure (psia)",
"sensor7" : "HPC Outlet Pressure (psia)",
"sensor8" : "Physical Fan Speed (rpm)",
"sensor9" : "Physical Core Speed (rpm)",
"sensor10" : "Engine Pressure Ratio(P50/P2)",
"sensor11" : "HPC Outlet Static Pressure (psia)",
"sensor12" : "Ratio of Fuel Flow to Ps30 (pps/psia)",
"sensor13" : "Corrected Fan Speed (rpm)",
"sensor14" : "Corrected Core Speed (rpm)",
"sensor15" : "Bypass Ratio",
"sensor16" : "Burner Fuel-Air Ratio",
"sensor17" : "Bleed Enthalpy",
"sensor18" : "Required Fan Speed (rpm)",
"sensor19" : "Required Fan Conversion Speed (rpm)",
"sensor20" : "High-Pressure Turbines Cool Air Flow (lbm/s)",
"sensor21" : "Low-Pressure Turbines Cool Air Flow (lbm/s)"
}

engine_columns = list(index_names.keys()) + list(setting_names.keys()) + list(sensor_names.keys())

# Load in each training engine data
comp_engine03_df = pd.read_csv(f"{data_directory}/train_FD003.txt", sep="\s+", header=None, names=engine_columns).drop(columns=['setting1', 'setting2', 'setting3'])

# Load in each testing engine data
test_engine03_df = pd.read_csv(f"{data_directory}/test_FD003.txt", sep="\s+", header=None, names=engine_columns).drop(columns=['setting1', 'setting2', 'setting3'])

# Load in each RUL engine data
rul_engine03_df = pd.read_csv(f"{data_directory}/RUL_FD003.txt", sep="\s+", header=None, names=['rul'])

# Define Feature Sets
numeric_features = comp_engine03_df.drop(columns=not_unique_col(comp_engine03_df)+["engine", 'cycle', 'sensor10']).columns.tolist()
categorical_features = []#comp_engine01_df[["engine", 'cycle']].columns.tolist()