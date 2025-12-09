import time
import numpy as np
from string import punctuation
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import sqlite3
from math import ceil
from collections import Counter
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.datasets import load_diabetes, load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score, root_mean_squared_error, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tensorflow as tf
from tensorflow import keras
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import timeit
import datetime
import pickle