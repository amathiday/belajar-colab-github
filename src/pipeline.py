"""Pipeline data: baca, bersihkan, dan siapkan model regresi penjualan."""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FITUR = ["biaya_iklan", "jumlah_promo"]
TARGET = "penjualan"


def muat_data(path):
    return pd.read_csv(path)


def bersihkan_data(df):
    # buang baris kosong dan angka negatif (data salah input)
    df = df.dropna(subset=FITUR + [TARGET])
    df = df[(df[FITUR] >= 0).all(axis=1)]
    return df.reset_index(drop=True)


def buat_model():
    return Pipeline([
        ("skala", StandardScaler()),
        ("regresi", LinearRegression()),
    ])
