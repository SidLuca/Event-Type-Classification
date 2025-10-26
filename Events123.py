# Phase 1: Data Understanding and Preparation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

# Step 1: Load and Inspect the Data
print("Step 1: Loading and inspecting data...")
# Load the CSV file
df = pd.read_csv('Events.csv')  # Replace with your actual file path

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names and data types:")
print(df.info())

print("\nBasic statistics:")
print(df.describe())

# Check for the target variable 'type'
print(f"\nUnique values in 'type' column: {sorted(df['type'].unique())}")

# Step 2: Analyze the Target Variable (Class Distribution)
print("\nStep 2: Analyzing class distribution...")
# Map the type codes to meaningful names for plotting
type_names = {
    0: 'B+B-',
    1: 'B0B0bar',
    2: 'ccbar',
    3: 'uubar',
    4: 'ddbar',
    5: 'ssbar'
}

# Create count plot for the multi-class distribution
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
type_counts = df['type'].value_counts().sort_index()
type_counts_named = [type_names[i] for i in type_counts.index]
plt.bar(type_counts_named, type_counts.values)
plt.title('Multi-Class Distribution (Original Types)')
plt.xlabel('Event Type')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Step 3: Create Binary Classification Target
print("\nStep 3: Creating binary classification target...")
df['isBBar'] = df['type'].apply(lambda x: 1 if x in [0, 1] else 0)

plt.subplot(1, 2, 2)
binary_counts = df['isBBar'].value_counts()
plt.bar(['Continuum (0)', 'BBbar (1)'], binary_counts.values)
plt.title('Binary Class Distribution (BBbar vs Continuum)')
plt.ylabel('Count')

for i, count in enumerate(binary_counts.values):
    plt.text(i, count + 1000, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Print detailed class distribution information
print("\nMulti-class distribution:")
for type_val, count in type_counts.items():
    print(f"{type_names[type_val]} (Type {type_val}): {count} events ({count / len(df) * 100:.2f}%)")

print(f"\nBinary distribution:")
print(f"BBbar events (isBBar=1): {binary_counts[1]} events ({binary_counts[1] / len(df) * 100:.2f}%)")
print(f"Continuum events (isBBar=0): {binary_counts[0]} events ({binary_counts[0] / len(df) * 100:.2f}%)")

# Step 4: Handle Missing and Invalid Values
print("\nStep 4: Handling missing and invalid values...")

# Check for missing values
print("Missing values in each column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Check for infinite values
print("\nChecking for infinite values...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
inf_mask = np.isinf(df[numeric_cols]).any()
if inf_mask.any():
    print("Columns with infinite values:")
    print(inf_mask[inf_mask].index.tolist())
else:
    print("No infinite values found.")

# Check for placeholder values (like -999)
print("\nChecking for potential placeholder values (-999)...")
placeholder_check = (df[numeric_cols] == -999).any()
if placeholder_check.any():
    print("Columns with -999 values:")
    print(placeholder_check[placeholder_check].index.tolist())
else:
    print("No -999 placeholder values found.")

# Handle missing values - remove rows with any missing values
initial_shape = df.shape
df_clean = df.dropna()
print(f"\nRemoved {initial_shape[0] - df_clean.shape[0]} rows with missing values")
print(f"Clean dataset shape: {df_clean.shape}")

# If there were infinite values, we would handle them here
# df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()

# Use the clean dataset for further analysis
df = df_clean

# Phase 2: Exploratory Data Analysis (EDA) & Feature Reduction

print("\n" + "=" * 50)
print("PHASE 2: Exploratory Data Analysis & Feature Reduction")
print("=" * 50)

# Step 5: Initial Feature Correlation Analysis
print("\nStep 5: Feature correlation analysis...")

# Get all feature columns (exclude target columns)
feature_columns = [col for col in df.columns if col not in ['type', 'isBBar']]
print(f"Number of feature columns: {len(feature_columns)}")

# Calculate correlation matrix
correlation_matrix = df[feature_columns].corr()

# Find highly correlated feature pairs
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.95:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

print(f"\nFound {len(high_corr_pairs)} highly correlated feature pairs (|correlation| > 0.95):")
for feat1, feat2, corr in high_corr_pairs[:10]:  # Show first 10
    print(f"  {feat1} <-> {feat2}: {corr:.3f}")

if len(high_corr_pairs) > 10:
    print(f"  ... and {len(high_corr_pairs) - 10} more pairs")

# Plot correlation heatmap for a subset of features (first 20 to avoid huge plot)
plt.figure(figsize=(15, 12))
subset_features = feature_columns[:20]
sns.heatmap(df[subset_features].corr(), cmap='coolwarm', center=0,
            annot=True, fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap (First 20 Features)')
plt.tight_layout()
plt.show()

# Step 6: Physics-Informed EDA
print("\nStep 6: Physics-informed exploratory data analysis...")

# Select key physics features for visualization
key_physics_features = [
    'R2', 'thrust', 'thrustBm', 'thrustOm', 'cosTBTO', 'cosTBz',
    'foxWolframR1', 'foxWolframR2', 'foxWolframR3', 'foxWolframR4'
]

# Only use features that actually exist in our dataset
available_physics_features = [f for f in key_physics_features if f in df.columns]
print(f"Available physics features for EDA: {available_physics_features}")

# Plot distributions of key physics features for binary classification
n_features = len(available_physics_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

plt.figure(figsize=(15, 4 * n_rows))
for i, feature in enumerate(available_physics_features, 1):
    plt.subplot(n_rows, n_cols, i)

    # Plot distributions for BBbar vs Continuum
    for target_val, color, label in [(1, 'red', 'BBbar'), (0, 'blue', 'Continuum')]:
        data = df[df['isBBar'] == target_val][feature]
        plt.hist(data, bins=50, alpha=0.6, color=color, label=label, density=True)

    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.title(f'Distribution of {feature}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Step 7: Univariate Feature Selection
print("\nStep 7: Univariate feature selection...")

# Prepare data for feature selection
X = df[feature_columns]
y_binary = df['isBBar']
y_multiclass = df['type']

# For binary classification task
print("\nBinary classification feature selection (isBBar vs Continuum):")
selector_binary = SelectKBest(score_func=f_classif, k='all')
selector_binary.fit(X, y_binary)

# Get feature scores for binary classification
binary_scores = pd.DataFrame({
    'feature': feature_columns,
    'f_score': selector_binary.scores_,
    'p_value': selector_binary.pvalues_
})
binary_scores = binary_scores.sort_values('f_score', ascending=False)

print("\nTop 20 features for binary classification (by F-score):")
print(binary_scores.head(20)[['feature', 'f_score', 'p_value']])

# For multi-class classification task
print("\nMulti-class classification feature selection:")
selector_multi = SelectKBest(score_func=f_classif, k='all')
selector_multi.fit(X, y_multiclass)

# Get feature scores for multi-class classification
multi_scores = pd.DataFrame({
    'feature': feature_columns,
    'f_score': selector_multi.scores_,
    'p_value': selector_multi.pvalues_
})
multi_scores = multi_scores.sort_values('f_score', ascending=False)

print("\nTop 20 features for multi-class classification (by F-score):")
print(multi_scores.head(20)[['feature', 'f_score', 'p_value']])

# Step 8: Model-Based Feature Selection
print("\nStep 8: Model-based feature selection...")

# Use a subset of data for faster feature importance calculation
sample_size = min(10000, len(df))
df_sample = df.sample(sample_size, random_state=42)

X_sample = df_sample[feature_columns]
y_sample_binary = df_sample['isBBar']

print(f"Using sample of {sample_size} events for feature importance calculation...")

# Train Random Forest for feature importance
rf_binary = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced',  # Important for imbalanced data
    n_jobs=-1
)

rf_binary.fit(X_sample, y_sample_binary)

# Get feature importances
feature_importances = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_binary.feature_importances_
})
feature_importances = feature_importances.sort_values('importance', ascending=False)

print("\nTop 25 features by Random Forest importance (Binary classification):")
print(feature_importances.head(25))

# Plot feature importances
plt.figure(figsize=(12, 8))
top_features = feature_importances.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 20 Features by Random Forest Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Create a reduced feature set based on feature importance
top_k_features = 25
selected_features = feature_importances.head(top_k_features)['feature'].tolist()

print(f"\nSelected top {top_k_features} features for model training:")
for i, feat in enumerate(selected_features, 1):
    print(f"{i:2d}. {feat}")

# Summary of Phase 2
print("\n" + "=" * 50)
print("PHASE 2 SUMMARY")
print("=" * 50)
print(f"Original number of features: {len(feature_columns)}")
print(f"Selected number of features: {len(selected_features)}")
print(f"Feature reduction: {len(feature_columns) - len(selected_features)} features removed")
print(f"Reduction percentage: {(len(feature_columns) - len(selected_features)) / len(feature_columns) * 100:.1f}%")

# Save the selected features for Phase 3
feature_selection_results = {
    'selected_features': selected_features,
    'feature_importances': feature_importances,
    'binary_scores': binary_scores,
    'multi_scores': multi_scores,
    'high_corr_pairs': high_corr_pairs
}

# Phase 3: Model Building and Training

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve, average_precision_score, roc_curve)

# Try to import XGBoost, handle if not installed
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not installed. Skipping XGBoost models.")
    XGBOOST_AVAILABLE = False

print("\n" + "=" * 50)
print("PHASE 3: Model Building and Training")
print("=" * 50)

# Redefine type_names for Phase 3 (in case running separately)
type_names = {
    0: 'B+B-',
    1: 'B0B0bar',
    2: 'ccbar',
    3: 'uubar',
    4: 'ddbar',
    5: 'ssbar'
}

# Step 9: Data Splitting and Preprocessing
print("\nStep 9: Data splitting and preprocessing...")

# Use the selected features from Phase 2
X = df[selected_features]
y_binary = df['isBBar']
y_multiclass = df['type']

print(f"Feature matrix shape: {X.shape}")
print(f"Binary target distribution: {np.bincount(y_binary)}")
print(f"Multi-class target distribution: {np.bincount(y_multiclass)}")

# Split data for binary classification
X_temp, X_test_bin, y_temp, y_test_bin = train_test_split(
    X, y_binary, test_size=0.15, random_state=42, stratify=y_binary
)
X_train_bin, X_val_bin, y_train_bin, y_val_bin = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.15/0.85 ≈ 0.176
)

# Split data for multi-class classification
X_temp, X_test_multi, y_temp, y_test_multi = train_test_split(
    X, y_multiclass, test_size=0.15, random_state=42, stratify=y_multiclass
)
X_train_multi, X_val_multi, y_train_multi, y_val_multi = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print("\nData splits for Binary Classification:")
print(f"Training set: {X_train_bin.shape[0]} events")
print(f"Validation set: {X_val_bin.shape[0]} events")
print(f"Test set: {X_test_bin.shape[0]} events")

print("\nData splits for Multi-class Classification:")
print(f"Training set: {X_train_multi.shape[0]} events")
print(f"Validation set: {X_val_multi.shape[0]} events")
print(f"Test set: {X_test_multi.shape[0]} events")

# Scale the features
print("\nScaling features...")
scaler_bin = StandardScaler()
scaler_multi = StandardScaler()

# Fit scalers on training data only
X_train_bin_scaled = scaler_bin.fit_transform(X_train_bin)
X_val_bin_scaled = scaler_bin.transform(X_val_bin)
X_test_bin_scaled = scaler_bin.transform(X_test_bin)

X_train_multi_scaled = scaler_multi.fit_transform(X_train_multi)
X_val_multi_scaled = scaler_multi.transform(X_val_multi)
X_test_multi_scaled = scaler_multi.transform(X_test_multi)

print("Feature scaling completed!")

# Step 10: Address Class Imbalance
print("\nStep 10: Addressing class imbalance...")

# Calculate class weights for binary classification
bin_class_weight = len(y_train_bin) / (2 * np.bincount(y_train_bin))
bin_class_weights = {0: bin_class_weight[0], 1: bin_class_weight[1]}
print(f"Binary class weights: {bin_class_weights}")

# Calculate class weights for multi-class classification
multi_class_weights = len(y_train_multi) / (len(np.unique(y_train_multi)) * np.bincount(y_train_multi))
multi_class_weight_dict = {i: weight for i, weight in enumerate(multi_class_weights)}
print(f"Multi-class class weights: {multi_class_weight_dict}")

# Step 11: Train Baseline Models
print("\nStep 11: Training baseline models...")

# Define models for binary classification
binary_models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    ),
    'Random Forest': RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(64, 32),
        random_state=42,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1
    )
}

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    binary_models['XGBoost'] = XGBClassifier(
        scale_pos_weight=len(y_train_bin[y_train_bin == 0]) / len(y_train_bin[y_train_bin == 1]),
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False
    )

# Define models for multi-class classification
multi_models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    ),
    'Random Forest': RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(64, 32),
        random_state=42,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1
    )
}

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    multi_models['XGBoost'] = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        use_label_encoder=False
    )

# Train and evaluate binary classification models
print("\n" + "=" * 60)
print("BINARY CLASSIFICATION RESULTS (Validation Set)")
print("=" * 60)

binary_results = {}

for name, model in binary_models.items():
    print(f"\n--- Training {name} ---")

    try:
        # Train the model
        if name in ['Logistic Regression', 'Neural Network']:
            model.fit(X_train_bin_scaled, y_train_bin)
        else:
            model.fit(X_train_bin, y_train_bin)

        # Make predictions
        if name in ['Logistic Regression', 'Neural Network']:
            y_pred = model.predict(X_val_bin_scaled)
            y_pred_proba = model.predict_proba(X_val_bin_scaled)[:, 1]
        else:
            y_pred = model.predict(X_val_bin)
            y_pred_proba = model.predict_proba(X_val_bin)[:, 1]

        # Calculate metrics
        auc_roc = roc_auc_score(y_val_bin, y_pred_proba)
        auc_pr = average_precision_score(y_val_bin, y_pred_proba)

        # Store results
        binary_results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr
        }

        print(f"ROC-AUC: {auc_roc:.4f}")
        print(f"PR-AUC: {auc_pr:.4f}")
        print("Classification Report:")
        print(classification_report(y_val_bin, y_pred, target_names=['Continuum', 'BBbar']))

    except Exception as e:
        print(f"Error training {name}: {str(e)}")
        continue

# Train and evaluate multi-class classification models
print("\n" + "=" * 60)
print("MULTI-CLASS CLASSIFICATION RESULTS (Validation Set)")
print("=" * 60)

multi_results = {}

for name, model in multi_models.items():
    print(f"\n--- Training {name} ---")

    try:
        # Train the model
        if name in ['Logistic Regression', 'Neural Network']:
            model.fit(X_train_multi_scaled, y_train_multi)
        else:
            model.fit(X_train_multi, y_train_multi)

        # Make predictions
        if name in ['Logistic Regression', 'Neural Network']:
            y_pred = model.predict(X_val_multi_scaled)
        else:
            y_pred = model.predict(X_val_multi)

        # Store results
        multi_results[name] = {
            'model': model,
            'y_pred': y_pred
        }

        print("Classification Report:")
        print(classification_report(y_val_multi, y_pred,
                                    target_names=[type_names[i] for i in range(6)]))

    except Exception as e:
        print(f"Error training {name}: {str(e)}")
        continue

# Visualization of Results
print("\nStep 12: Visualizing model performance...")

# Plot ROC curves for binary classification
plt.figure(figsize=(15, 10))

# FIXED: Use roc_curve instead of precision_recall_curve
plt.subplot(2, 3, 1)
for name, results in binary_results.items():
    fpr, tpr, _ = roc_curve(y_val_bin, results['y_pred_proba'])  # FIXED
    plt.plot(fpr, tpr, label=f'{name} (AUC = {results["auc_roc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Binary Classification')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Precision-Recall curves for binary classification
plt.subplot(2, 3, 2)
for name, results in binary_results.items():
    precision, recall, _ = precision_recall_curve(y_val_bin, results['y_pred_proba'])
    plt.plot(recall, precision, label=f'{name} (AP = {results["auc_pr"]:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves - Binary Classification')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot confusion matrix for best binary model
plt.subplot(2, 3, 3)
best_binary_model = max(binary_results.items(), key=lambda x: x[1]['auc_roc'])[0]
y_pred_best = binary_results[best_binary_model]['y_pred']
cm = confusion_matrix(y_val_bin, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Continuum', 'BBbar'],
            yticklabels=['Continuum', 'BBbar'])
plt.title(f'Confusion Matrix - {best_binary_model}\n(Binary Classification)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Plot confusion matrix for best multi-class model
plt.subplot(2, 3, 4)
# Find best multi-class model (using macro F1 as proxy)
best_multi_model = None
best_multi_score = 0
for name, results in multi_results.items():
    report = classification_report(y_val_multi, results['y_pred'], output_dict=True)
    macro_f1 = report['macro avg']['f1-score']
    if macro_f1 > best_multi_score:
        best_multi_score = macro_f1
        best_multi_model = name

y_pred_best_multi = multi_results[best_multi_model]['y_pred']
cm_multi = confusion_matrix(y_val_multi, y_pred_best_multi)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=[type_names[i] for i in range(6)],
            yticklabels=[type_names[i] for i in range(6)])
plt.title(f'Confusion Matrix - {best_multi_model}\n(Multi-class Classification)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Plot model comparison for binary classification
plt.subplot(2, 3, 5)
models_bin = list(binary_results.keys())
auc_scores_bin = [binary_results[name]['auc_roc'] for name in models_bin]
ap_scores_bin = [binary_results[name]['auc_pr'] for name in models_bin]

x = np.arange(len(models_bin))
width = 0.35

plt.bar(x - width / 2, auc_scores_bin, width, label='ROC-AUC', alpha=0.8)
plt.bar(x + width / 2, ap_scores_bin, width, label='PR-AUC', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Binary Classification Performance Comparison')
plt.xticks(x, models_bin, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot feature importance for the best model
plt.subplot(2, 3, 6)
if hasattr(binary_results[best_binary_model]['model'], 'feature_importances_'):
    importances = binary_results[best_binary_model]['model'].feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': selected_features,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(10)

    plt.barh(feature_imp_df['feature'], feature_imp_df['importance'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 10 Features - {best_binary_model}')
else:
    # For linear models, use absolute coefficients
    if hasattr(binary_results[best_binary_model]['model'], 'coef_'):
        coef = binary_results[best_binary_model]['model'].coef_[0]
        feature_imp_df = pd.DataFrame({
            'feature': selected_features,
            'importance': np.abs(coef)
        }).sort_values('importance', ascending=True).tail(10)

        plt.barh(feature_imp_df['feature'], feature_imp_df['importance'])
        plt.xlabel('Absolute Coefficient')
        plt.title(f'Top 10 Features - {best_binary_model}')
    else:
        plt.text(0.5, 0.5, 'Feature importance\nnot available',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Importance')

plt.tight_layout()
plt.show()

# Summary of Results
print("\n" + "=" * 60)
print("PHASE 3 SUMMARY")
print("=" * 60)

print("\nBINARY CLASSIFICATION - Best Model:", best_binary_model)
best_binary_metrics = binary_results[best_binary_model]
print(f"ROC-AUC: {best_binary_metrics['auc_roc']:.4f}")
print(f"PR-AUC: {best_binary_metrics['auc_pr']:.4f}")

print("\nMULTI-CLASS CLASSIFICATION - Best Model:", best_multi_model)
print(f"Macro F1-Score: {best_multi_score:.4f}")

print("\nAll Binary Classification Results (sorted by ROC-AUC):")
print("-" * 50)
for name, score in sorted(binary_results.items(), key=lambda x: x[1]['auc_roc'], reverse=True):
    print(f"{name:20} | ROC-AUC: {score['auc_roc']:.4f} | PR-AUC: {score['auc_pr']:.4f}")

# Save the trained models and results for Phase 4
phase3_results = {
    'binary_models': binary_models,
    'multi_models': multi_models,
    'binary_results': binary_results,
    'multi_results': multi_results,
    'best_binary_model': best_binary_model,
    'best_multi_model': best_multi_model,
    'scaler_bin': scaler_bin,
    'scaler_multi': scaler_multi,
    'selected_features': selected_features,
    'type_names': type_names
}

print("\nPhase 3 completed successfully!")
print("Key outputs saved for Phase 4: Hyperparameter Tuning and Final Evaluation")
# Phase 4: Hyperparameter Tuning and Final Evaluation

from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import roc_curve, precision_recall_curve, classification_report, confusion_matrix
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

print("=" * 50)
print("PHASE 4: Hyperparameter Tuning and Final Evaluation")
print("=" * 50)

# Load results from Phase 3
# phase3_results should contain: best_binary_model, binary_models, scaler_bin, selected_features, etc.

# Step 12: Hyperparameter Tuning
print("\nStep 12: Hyperparameter Tuning...")

# Identify the best model from Phase 3 for tuning
best_model_name = phase3_results['best_binary_model']
print(f"Tuning the best model from Phase 3: {best_model_name}")

# Define hyperparameter grids for different models
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample']
    },

    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    },

    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced', None]
    },

    'Neural Network': {
        'hidden_layer_sizes': [(64, 32), (128, 64), (64, 64, 32), (128, 64, 32)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
        'early_stopping': [True]
    }
}

# Select the appropriate parameter grid
if best_model_name in param_grids:
    param_grid = param_grids[best_model_name]
else:
    print(f"No parameter grid defined for {best_model_name}. Using default parameters.")
    param_grid = {}

# Perform hyperparameter tuning if grid exists
if param_grid:
    print(f"Performing RandomizedSearchCV for {best_model_name}...")

    # Get the base model
    base_model = phase3_results['binary_models'][best_model_name]

    # Set up randomized search
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter combinations to try
        cv=3,  # 3-fold cross-validation
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    # Fit the randomized search
    start_time = time.time()

    if best_model_name in ['Logistic Regression', 'Neural Network']:
        random_search.fit(X_train_bin_scaled, y_train_bin)
    else:
        random_search.fit(X_train_bin, y_train_bin)

    end_time = time.time()
    print(f"Hyperparameter tuning completed in {end_time - start_time:.2f} seconds")

    # Get the best model
    best_tuned_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print(f"\nBest parameters for {best_model_name}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best cross-validation ROC-AUC: {best_score:.4f}")

else:
    print("Using baseline model without hyperparameter tuning")
    best_tuned_model = phase3_results['binary_results'][best_model_name]['model']
    best_params = "Default parameters"
    best_score = phase3_results['binary_results'][best_model_name]['auc_roc']

# Step 13: Final Evaluation on Test Set
print("\nStep 13: Final Evaluation on Test Set...")

# Prepare test data
if best_model_name in ['Logistic Regression', 'Neural Network']:
    X_test_final = X_test_bin_scaled
else:
    X_test_final = X_test_bin

# Make predictions on test set
if hasattr(best_tuned_model, 'predict_proba'):
    y_test_pred_proba = best_tuned_model.predict_proba(X_test_final)[:, 1]
    y_test_pred = best_tuned_model.predict(X_test_final)
else:
    y_test_pred = best_tuned_model.predict(X_test_final)
    y_test_pred_proba = None

# Calculate comprehensive metrics
test_auc_roc = roc_auc_score(y_test_bin, y_test_pred_proba) if y_test_pred_proba is not None else None
test_auc_pr = average_precision_score(y_test_bin, y_test_pred_proba) if y_test_pred_proba is not None else None

print("\n" + "=" * 60)
print("FINAL TEST SET PERFORMANCE")
print("=" * 60)
print(f"Model: {best_model_name}")
if test_auc_roc:
    print(f"ROC-AUC: {test_auc_roc:.4f}")
    print(f"PR-AUC: {test_auc_pr:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_bin, y_test_pred, target_names=['Continuum', 'BBbar']))

# Calculate physics-style metrics
cm_test = confusion_matrix(y_test_bin, y_test_pred)
tn, fp, fn, tp = cm_test.ravel()

signal_efficiency = tp / (tp + fn)  # Recall for BBbar
background_rejection = tn / (tn + fp)  # Specificity for Continuum
purity = tp / (tp + fp)  # Precision for BBbar

print(f"\nPhysics Metrics:")
print(f"Signal Efficiency (BBbar): {signal_efficiency:.4f}")
print(f"Background Rejection (Continuum): {background_rejection:.4f}")
print(f"Purity (BBbar): {purity:.4f}")

# Step 14: Model Interpretation and Physics Validation
print("\nStep 14: Model Interpretation and Physics Validation...")

# Feature Importance Analysis
print("\nFeature Importance Analysis:")
if hasattr(best_tuned_model, 'feature_importances_'):
    importances = best_tuned_model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': selected_features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Most Important Features:")
    for i, row in feature_imp_df.head(15).iterrows():
        print(f"  {row['feature']:30} : {row['importance']:.4f}")

elif hasattr(best_tuned_model, 'coef_'):
    # For linear models
    coef = best_tuned_model.coef_[0]
    feature_imp_df = pd.DataFrame({
        'feature': selected_features,
        'importance': np.abs(coef)
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Most Important Features (Absolute Coefficients):")
    for i, row in feature_imp_df.head(15).iterrows():
        print(f"  {row['feature']:30} : {row['importance']:.4f}")

# Analyze high-signal region
if y_test_pred_proba is not None:
    print(f"\nHigh-Signal Region Analysis:")

    # Define high-signal threshold (top 10% of predicted probabilities)
    high_signal_threshold = np.percentile(y_test_pred_proba, 90)
    high_signal_mask = y_test_pred_proba >= high_signal_threshold

    high_signal_events = y_test_bin[high_signal_mask]
    high_signal_purity = np.mean(high_signal_events)  # Fraction that are truly BBbar

    print(f"Events in high-signal region (top 10%): {len(high_signal_events)}")
    print(f"Purity in high-signal region: {high_signal_purity:.4f}")
    print(f"Signal efficiency in high-signal region: {np.sum(high_signal_events) / np.sum(y_test_bin):.4f}")

# Comprehensive Visualizations
print("\nGenerating comprehensive visualizations...")

plt.figure(figsize=(20, 15))

# 1. ROC Curve
plt.subplot(3, 3, 1)
if y_test_pred_proba is not None:
    fpr, tpr, _ = roc_curve(y_test_bin, y_test_pred_proba)
    plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {test_auc_roc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate (Background)')
    plt.ylabel('True Positive Rate (Signal Efficiency)')
    plt.title('ROC Curve - Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)

# 2. Precision-Recall Curve
plt.subplot(3, 3, 2)
if y_test_pred_proba is not None:
    precision, recall, _ = precision_recall_curve(y_test_bin, y_test_pred_proba)
    plt.plot(recall, precision, label=f'{best_model_name} (AP = {test_auc_pr:.3f})', linewidth=2)
    plt.xlabel('Recall (Signal Efficiency)')
    plt.ylabel('Precision (Purity)')
    plt.title('Precision-Recall Curve - Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)

# 3. Confusion Matrix
plt.subplot(3, 3, 3)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Continuum', 'BBbar'],
            yticklabels=['Continuum', 'BBbar'])
plt.title('Confusion Matrix - Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 4. Feature Importance
plt.subplot(3, 3, 4)
if 'feature_imp_df' in locals():
    top_features = feature_imp_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importances')
    plt.gca().invert_yaxis()

# 5. Probability Distributions
plt.subplot(3, 3, 5)
if y_test_pred_proba is not None:
    for target_val, color, label in [(1, 'red', 'BBbar (Signal)'), (0, 'blue', 'Continuum (Background)')]:
        mask = y_test_bin == target_val
        plt.hist(y_test_pred_proba[mask], bins=50, alpha=0.6,
                 color=color, label=label, density=True)
    plt.xlabel('Predicted Probability (BBbar)')
    plt.ylabel('Density')
    plt.title('Predicted Probability Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)

# 6. Background Rejection vs Signal Efficiency
plt.subplot(3, 3, 6)
if y_test_pred_proba is not None:
    fpr, tpr, thresholds = roc_curve(y_test_bin, y_test_pred_proba)
    background_rejection_curve = 1 - fpr

    plt.plot(tpr, background_rejection_curve, linewidth=2)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Rejection')
    plt.title('Background Rejection vs Signal Efficiency')
    plt.grid(True, alpha=0.3)

    # Mark the operating point
    plt.plot(signal_efficiency, background_rejection, 'ro', markersize=8,
             label=f'Operating Point\nEff: {signal_efficiency:.3f}\nRej: {background_rejection:.3f}')
    plt.legend()

# 7. Score Distribution by True Type
plt.subplot(3, 3, 7)
if y_test_pred_proba is not None:
    plt.hist(y_test_pred_proba[y_test_bin == 1], bins=50, alpha=0.7,
             label='True BBbar', color='red', density=True)
    plt.hist(y_test_pred_proba[y_test_bin == 0], bins=50, alpha=0.7,
             label='True Continuum', color='blue', density=True)
    plt.xlabel('Classifier Score')
    plt.ylabel('Density')
    plt.title('Classifier Score Distribution by True Type')
    plt.legend()
    plt.grid(True, alpha=0.3)

# 8. Performance Summary
plt.subplot(3, 3, 8)
metrics = ['ROC-AUC', 'PR-AUC', 'Signal Eff', 'Bgnd Rej', 'Purity']
values = [test_auc_roc or 0, test_auc_pr or 0, signal_efficiency, background_rejection, purity]
colors = ['blue', 'green', 'red', 'orange', 'purple']

bars = plt.bar(metrics, values, color=colors, alpha=0.7)
plt.ylim(0, 1)
plt.title('Performance Metrics Summary')
plt.xticks(rotation=45)

# Add value labels on bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

# 9. Model Comparison (Baseline vs Tuned)
plt.subplot(3, 3, 9)
if 'best_score' in locals() and test_auc_roc:
    comparison_data = {
        'Baseline': phase3_results['binary_results'][best_model_name]['auc_roc'],
        'Tuned': test_auc_roc
    }
    plt.bar(comparison_data.keys(), comparison_data.values(), color=['lightblue', 'lightgreen'])
    plt.ylabel('ROC-AUC')
    plt.title('Baseline vs Tuned Performance')
    for i, (name, value) in enumerate(comparison_data.items()):
        plt.text(i, value + 0.01, f'{value:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Save the final model and results
print("\nSaving final model and results...")
final_results = {
    'best_model': best_tuned_model,
    'best_model_name': best_model_name,
    'best_params': best_params,
    'test_metrics': {
        'roc_auc': test_auc_roc,
        'pr_auc': test_auc_pr,
        'signal_efficiency': signal_efficiency,
        'background_rejection': background_rejection,
        'purity': purity
    },
    'feature_importances': feature_imp_df if 'feature_imp_df' in locals() else None,
    'selected_features': selected_features,
    'scaler': scaler_bin
}

# Save model to file
model_filename = f'final_{best_model_name.replace(" ", "_").lower()}_model.pkl'
joblib.dump(best_tuned_model, model_filename)
print(f"Final model saved as: {model_filename}")

# Save results to file
results_filename = 'phase4_final_results.pkl'
joblib.dump(final_results, results_filename)
print(f"Final results saved as: {results_filename}")

# Final Summary
print("\n" + "=" * 60)
print("PHASE 4 COMPLETE - FINAL SUMMARY")
print("=" * 60)
print(f"Best Model: {best_model_name}")
print(f"Test ROC-AUC: {test_auc_roc:.4f}" if test_auc_roc else "Test ROC-AUC: N/A")
print(f"Test PR-AUC: {test_auc_pr:.4f}" if test_auc_pr else "Test PR-AUC: N/A")
print(f"Signal Efficiency: {signal_efficiency:.4f}")
print(f"Background Rejection: {background_rejection:.4f}")
print(f"Purity: {purity:.4f}")
print(f"\nModel successfully trained and evaluated on {X_test_final.shape[0]} test events")
print("Ready for deployment or further analysis!")