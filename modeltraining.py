import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, warnings
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

df = pd.read_csv('Events.csv')
if 'type' not in df.columns:
    raise ValueError("Dataset must contain a 'type' column as target.")

df['isBBar'] = df['type'].apply(lambda x: 1 if x in [0, 1] else 0)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

features = [c for c in df.columns if c not in ['type', 'isBBar']]
X = df[features]
y = df['isBBar']

imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=features)

corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]
if to_drop:
    X = X.drop(columns=to_drop)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)
perm = permutation_importance(rf, X, y, n_repeats=5, random_state=42, n_jobs=-1)
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm.importances_mean
}).sort_values('Importance', ascending=False).head(25)

selected_features = importances['Feature'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X[selected_features], y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
except ImportError:
    print("Install with: pip install xgboost lightgbm")
    raise

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=64,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42
    ),
    "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
}

def evaluate_model(name, model, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, cv):
    use_scaled = name in ["Logistic Regression", "Neural Network (MLP)", "SVM"]
    Xtr, Xte = (X_train_scaled, X_test_scaled) if use_scaled else (X_train, X_test)

    scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_mean, cv_std = scores.mean(), scores.std()

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    pr_auc = average_precision_score(y_test, y_proba) if y_proba is not None else np.nan

    return {
        "Model": name,
        "CV Mean AUC": cv_mean,
        "CV Std AUC": cv_std,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "model": model
    }

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(evaluate_model)(
        name, model, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, cv
    )
    for name, model in models.items()
)

results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_pred', 'y_proba', 'model']} for r in results])

plt.figure(figsize=(10, 6))
bar_width = 0.13
x = np.arange(len(results_df["Model"]))
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]

for i, metric in enumerate(metrics):
    plt.bar(x + i * bar_width, results_df[metric], width=bar_width, label=metric)

plt.xticks(x + bar_width * (len(metrics) - 1) / 2, results_df["Model"], rotation=25)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()

probabilities = {r["Model"]: (r["y_pred"], r["y_proba"]) for r in results}

plt.figure(figsize=(8, 6))
for name, (_, y_proba) in probabilities.items():
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
for name, (_, y_proba) in probabilities.items():
    if y_proba is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        auc = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, label=f"{name} (AUC={auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.tight_layout()
plt.show()

best_idx = results_df["ROC-AUC"].idxmax()
best_model_info = results[best_idx]
best_model_name = best_model_info["Model"]
best_model = best_model_info["model"]

save_dict = {
    "model": best_model,
    "scaler": scaler,
    "features": selected_features,
    "metrics": results_df.set_index("Model").loc[best_model_name].to_dict()
}
# save_path = f"best_model_{best_model_name.replace(' ', '_').lower()}_{pd.Timestamp.now().date()}.pkl"
#joblib.dump(save_dict, save_path)

#print(f"Best model saved: {best_model_name} at {save_path}")