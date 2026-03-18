import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

# ── Load data ──────────────────────────────────────────
df = pd.read_csv("D:/multi-disease-predictor/data/diabetes.csv")

# ── Check data ─────────────────────────────────────────
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Nulls:\n", df.isnull().sum())

# ── Preprocessing ──────────────────────────────────────
# In diabetes dataset 0 values in these columns mean missing
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── SVM Pipeline ───────────────────────────────────────
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf", probability=True, random_state=42))
])
svm_pipe.fit(X_train, y_train)

# ── Random Forest ──────────────────────────────────────
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ── XGBoost ────────────────────────────────────────────
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric="logloss")
xgb_model.fit(X_train, y_train)

# ── Evaluate all models ────────────────────────────────
models = {"SVM": svm_pipe, "Random Forest": rf_model, "XGBoost": xgb_model}

print("\n── Model Comparison ──────────────────────")
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n{name}:")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

# ── Save SVM model ─────────────────────────────────────
joblib.dump(svm_pipe, "D:/multi-disease-predictor/saved_models/diabetes_svm.pkl")
print("\n✓ Diabetes SVM model saved!")