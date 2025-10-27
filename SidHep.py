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
df = pd.read_csv('Events.csv')  

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


initial_shape = df.shape
df_clean = df.dropna()
print(f"\nRemoved {initial_shape[0] - df_clean.shape[0]} rows with missing values")
print(f"Clean dataset shape: {df_clean.shape}")


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


print("\nPhase 2 completed successfully! Ready for Phase 3: Model Building and Training.")
