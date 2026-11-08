# belajar-colab-github

Latihan menyimpan notebook Google Colab ke GitHub, plus contoh model regresi
penjualan dari development sampai production.

## Struktur

```
data/penjualan.csv   data biaya iklan, jumlah promo, dan penjualan
src/pipeline.py      baca, bersihkan data, dan susun model
src/train.py         development: latih model lalu simpan
src/predict.py       production: muat model lalu prediksi
models/model.joblib  model yang sudah dilatih
requirements.txt     library yang dibutuhkan
```

## Development

```
pip install -r requirements.txt
python src/train.py
```

## Production

Ambil versi rilis (tag), bukan branch yang sedang dikerjakan:

```
git clone --branch v1.0.0 https://github.com/amathiday/belajar-colab-github.git
cd belajar-colab-github
pip install -r requirements.txt
python src/predict.py 20 3
```
