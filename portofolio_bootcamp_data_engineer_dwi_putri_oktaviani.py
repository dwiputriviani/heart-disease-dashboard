import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Heart Disease Dashboard",
    layout="wide"
)

# =========================
# LOAD DATA
# ========
