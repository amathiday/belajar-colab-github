"""Development: latih model regresi lalu simpan ke models/model.joblib."""
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from pipeline import FITUR, TARGET, bersihkan_data, buat_model, muat_data

df = bersihkan_data(muat_data("data/penjualan.csv"))
X_train, X_test, y_train, y_test = train_test_split(
    df[FITUR], df[TARGET], test_size=0.2, random_state=42)

model = buat_model()
model.fit(X_train, y_train)

prediksi = model.predict(X_test)
print(f"Jumlah data bersih : {len(df)}")
print(f"R2                 : {r2_score(y_test, prediksi):.3f}")
print(f"MAE                : {mean_absolute_error(y_test, prediksi):.2f} jt")

joblib.dump(model, "models/model.joblib")
print("Model disimpan di models/model.joblib")
