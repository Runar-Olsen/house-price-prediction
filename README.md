# 🏠 House Price Prediction (Regression, Python)

Et end-to-end prosjekt som predikerer boligpriser (SalePrice) med **regresjonsmodeller** i Python.
Prosjektet viser:
- Databehandling med `ColumnTransformer` (imputering, skalering, one-hot)
- Sammenligning av modeller: Linear Regression, Random Forest, (valgfritt) XGBoost
- Evaluering med **RMSE**, **MAE** og **R²**
- Lagring av beste modell som `.joblib`

---

## 📦 Datasett
Kaggle: **House Prices – Advanced Regression Techniques**  
Legg nedlastet `train.csv` i: data/raw/train.csv

---

## 🛠️ Teknologier
Python · Pandas · NumPy · Scikit-learn · (XGBoost) · Matplotlib/Seaborn · Joblib

---

## 🗂️ Struktur
```text
house-price-prediction/
├─ data/
│  ├─ raw/            # train.csv her
│  └─ processed/
├─ src/
│  ├─ __init__.py
│  ├─ utils.py
│  └─ train_regression.py
├─ models/
├─ reports/
│  └─ figures/
├─ notebooks/
├─ .gitignore
├─ requirements.txt
└─ README.md
