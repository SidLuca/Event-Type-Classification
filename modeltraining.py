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

# Removed SVM from models dictionary
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
    )
    # SVM removed from here
}


def evaluate_model(name, model, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, cv):
    use_scaled = name in ["Logistic Regression", "Neural Network (MLP)"]  # Removed SVM from scaled models
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

# Add formatted table printout
print("\n" + "=" * 80)
print("MODEL PERFORMANCE COMPARISON TABLE")
print("=" * 80)

# Format the results for display
display_df = results_df.copy()
display_df = display_df.round(6)

# Reorder columns for better readability
column_order = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "CV Mean AUC", "CV Std AUC"]
display_df = display_df[column_order]

# Print the header
header = f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10} {'PR-AUC':<10}"
print(header)
print("-" * 95)

# Print each model's results
for _, row in display_df.iterrows():
    model_name = row['Model']
    # Truncate long model names for better formatting
    if len(model_name) > 24:
        model_name = model_name[:21] + "..."

    print(f"{model_name:<25} {row['Accuracy']:<10.6f} {row['Precision']:<10.6f} {row['Recall']:<10.6f} "
          f"{row['F1']:<10.6f} {row['ROC-AUC']:<10.6f} {row['PR-AUC']:<10.6f}")

print("-" * 95)

# Alternative: Using pandas built-in display with formatting
print("\nFormatted Results Table:")
print("-" * 95)
formatted_df = results_df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]].copy()
formatted_df = formatted_df.round(6)
print(formatted_df.to_string(index=False, formatters={
    'Accuracy': '{:.6f}'.format,
    'Precision': '{:.6f}'.format,
    'Recall': '{:.6f}'.format,
    'F1': '{:.6f}'.format,
    'ROC-AUC': '{:.6f}'.format,
    'PR-AUC': '{:.6f}'.format
}))
print("-" * 95)

# Identify and highlight the best model
best_roc_auc_model = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
best_f1_model = results_df.loc[results_df['F1'].idxmax(), 'Model']
best_accuracy_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']

print(f"\nBEST PERFORMING MODELS:")
print(f"  Highest ROC-AUC: {best_roc_auc_model} ({results_df['ROC-AUC'].max():.6f})")
print(f"  Highest F1-Score: {best_f1_model} ({results_df['F1'].max():.6f})")
print(f"  Highest Accuracy: {best_accuracy_model} ({results_df['Accuracy'].max():.6f})")

# Continue with plotting (adjusted for 5 models instead of 6)
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
# joblib.dump(save_dict, save_path)

# print(f"Best model saved: {best_model_name} at {save_path}")
