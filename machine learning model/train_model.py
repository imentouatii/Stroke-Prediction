import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("heart.csv", sep=";")

TARGET = "HeartDisease"
X = df.drop(columns=[TARGET])
y = df[TARGET]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# =====================
# PREPROCESSING
# =====================
num_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

cat_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ]
)

# =====================
# MODEL
# =====================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", rf),
    ]
)

# =====================
# TRAIN / TEST
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)

print(f"✅ RandomForest AUC = {auc:.4f}")

# =====================
# SAVE MODEL
# =====================
joblib.dump(model, "heart_guard_best.joblib")
print("💾 Modèle sauvegardé : heart_guard_best.joblib")
