![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Regression-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


# 🏠 House Price Prediction (Regression Models)

Et komplett maskinlæringsprosjekt som predikerer boligpriser (Ames Housing Dataset) ved hjelp av flere regresjonsmodeller:  
Linear Regression, Random Forest, Gradient Boosting m.m.

Prosjektet inkluderer dataforberedelse, modelltrening, lagring og evaluering.

---

## 🎯 Hva prosjektet demonstrerer

- Feature engineering og håndtering av manglende verdier  
- Standardisering av numeriske verdier  
- Sammenligning av flere regresjonsmodeller  
- Evaluering med MAE, MSE, RMSE og R²  
- Lagre modeller med joblib  
- Produksjonsklar folderstruktur

---

## 🗂️ Prosjektstruktur

```text
house-price-prediction/
├─ data/
│  └─ housing.csv
│
├─ models/
│  ├─ linear_regression.pkl
│  ├─ random_forest.pkl
│  └─ gradient_boosting.pkl
│
├─ reports/
│  └─ regression_scores.json
│
├─ src/
│  ├─ preprocess.py
│  ├─ train_regression.py
│  └─ utils.py
│
├─ requirements.txt
└─ README.md
```

---

## ▶️ Kom i gang
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train_regression
```

---

## 📊 Evalueringsmetrikker
- R² Score
- MAE - Mean Absolute Error
- MSE - Mean Squared Error
- RMSE - Root Mean Squared Error

---

# 🚀 Videre arbeid 
- Hyperparatemeter-tuning (GridSearchCV / Optuna)
- SHAP for model explainability
- Web-app for prediksjon med egendetinerte inputfelt

# 👤 Forfatter
- ### Runar Olsen
