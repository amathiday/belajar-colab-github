"""Production: muat model yang sudah dilatih lalu prediksi data baru."""
import sys

import joblib
import pandas as pd

from pipeline import FITUR

model = joblib.load("models/model.joblib")


def prediksi(biaya_iklan, jumlah_promo):
    data = pd.DataFrame([[biaya_iklan, jumlah_promo]], columns=FITUR)
    return float(model.predict(data)[0])


if __name__ == "__main__":
    iklan = float(sys.argv[1]) if len(sys.argv) > 1 else 20
    promo = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    hasil = prediksi(iklan, promo)
    print(f"Prediksi penjualan (iklan {iklan:g} jt, promo {promo}x): {hasil:.1f} jt")
