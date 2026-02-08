import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# XGBoost (optionnel)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

# =========================
# 1) Load data
# =========================
df = pd.read_csv("heart.csv", sep=";")
TARGET = "HeartDisease"

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# Colonnes
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# =========================
# 2) Preprocess
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# =========================
# 3) Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 4) Models
# =========================
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    ),
}

if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    )

# =========================
# 5) Train + Evaluate
# =========================
results = []
best_model_name = None
best_auc = -1
best_pipeline = None

for name, clf in models.items():
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", clf)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append((name, acc, f1, auc))

    print("\n==============================")
    print("Model:", name)
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    # Choix du meilleur modèle selon AUC
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_pipeline = pipe

# =========================
# 6) Résumé + save
# =========================
print("\n✅ Résumé comparaison :")
for r in results:
    print(f"- {r[0]} => Accuracy={r[1]:.4f}, F1={r[2]:.4f}, AUC={r[3]:.4f}")

print(f"\n🏆 Meilleur modèle = {best_model_name} (AUC={best_auc:.4f})")

joblib.dump(best_pipeline, "heart_guard_best.joblib")
print("✅ Modèle sauvegardé : heart_guard_best.joblib")
