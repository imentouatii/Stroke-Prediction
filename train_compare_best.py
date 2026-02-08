import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# XGBoost (optionnel)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# LightGBM (optionnel)
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# =========================
# 0) Paramètres
# =========================
USE_SMOTE = True   # mets False si tu ne veux pas SMOTE
RANDOM_STATE = 42
TARGET = "HeartDisease"


# =========================
# 1) Load data
# =========================
df = pd.read_csv("heart.csv", sep=";")
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# =========================
# 2) Preprocess
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# =========================
# 3) Train/Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# =========================
# 4) Modèles
# =========================
models = {
    "LogisticRegression": LogisticRegression(max_iter=3000),
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
}

if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

if HAS_LGBM:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
    )

# =========================
# 5) Train + Evaluate
# =========================
rows = []
best_name = None
best_auc = -1
best_pipe = None

for name, clf in models.items():
    if USE_SMOTE:
        pipe = ImbPipeline(steps=[
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", clf),
        ])
    else:
        pipe = Pipeline(steps=[
            ("preprocess", preprocess),
            ("clf", clf),
        ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    rows.append({"model": name, "accuracy": acc, "f1": f1, "auc": auc})

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # on choisit le meilleur selon AUC (le + robuste en classification)
    if auc > best_auc:
        best_auc = auc
        best_name = name
        best_pipe = pipe

# Résultats
results = pd.DataFrame(rows).sort_values("auc", ascending=False)
print("\n✅ Comparaison finale (tri AUC):")
print(results.to_string(index=False))

# =========================
# 6) Save best model
# =========================
joblib.dump(best_pipe, "heart_guard_best.joblib")
print(f"\n🏆 Meilleur modèle: {best_name} (AUC={best_auc:.4f})")
print("✅ Sauvegardé: heart_guard_best.joblib")

# =========================
# 7) Plot scores
# =========================
plt.figure()
plt.plot(results["model"], results["auc"], marker="o", label="AUC")
plt.plot(results["model"], results["f1"], marker="o", label="F1")
plt.plot(results["model"], results["accuracy"], marker="o", label="Accuracy")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 1)
plt.title("Comparaison des modèles")
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=200)
print("📊 Graphique sauvegardé: model_comparison.png")
