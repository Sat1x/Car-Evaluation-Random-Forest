import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Load and Explore Data ---
print("Loading car evaluation dataset...")
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data")
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names

print("\nDataset Head:")
print(df.head())
print("\nDataset Info:")
df.info()

# --- 2. Data Visualization ---
print("\nVisualizing data distributions...")

# Class distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='class', hue='class', order=df['class'].value_counts().index, palette='viridis', legend=False)
plt.title('Class Distribution in Car Evaluation Dataset', fontsize=16)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('github projects/car_eval_random_forest/class_distribution.png', bbox_inches='tight')
plt.show()

# --- 3. Preprocessing ---
print("\nEncoding categorical features...")
# Keep original dataframe for readable labels in plots
df_encoded = df.copy()
oe = OrdinalEncoder()
for col in df_encoded.columns:
    df_encoded[col] = oe.fit_transform(df_encoded[[col]])

# --- 4. Data Splitting ---
X = df_encoded.drop('class', axis=1)
y = df_encoded['class']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nData split into {len(x_train)} training and {len(x_test)} testing samples.")

# --- 5. Baseline Model ---
print("\nTraining baseline Random Forest model...")
rfc_base = RandomForestClassifier(random_state=42)
rfc_base.fit(x_train, y_train)
base_pred = rfc_base.predict(x_test)
print(f"Baseline Model Accuracy: {accuracy_score(y_test, base_pred):.4f}")

# --- 6. Hyperparameter Tuning Visualization (replaces yellowbrick) ---
print("\nVisualizing hyperparameter tuning impact...")

def plot_validation_curve_sk(param_name, param_range):
    """Plots validation curve for a given hyperparameter."""
    train_scores, test_scores = validation_curve(
        RandomForestClassifier(random_state=42), x_train, y_train,
        param_name=param_name, param_range=param_range, cv=3, scoring="accuracy", n_jobs=1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, label="Training score", color="darkorange")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="navy")
    plt.title(f'Validation Curve for {param_name}', fontsize=16)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f'github projects/car_eval_random_forest/validation_curve_{param_name}.png', bbox_inches='tight')
    plt.show()

# Plot for n_estimators
plot_validation_curve_sk('n_estimators', [50, 100, 200, 300, 400, 500])
# Plot for max_depth
plot_validation_curve_sk('max_depth', [5, 10, 15, 20, 25, 30])
# Plot for min_samples_split
plot_validation_curve_sk('min_samples_split', [2, 5, 10, 15, 20])

# --- 7. Tuned Model ---
print("\nTraining tuned Random Forest model...")
rfc_tuned = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=2, random_state=42)
rfc_tuned.fit(x_train, y_train)
tuned_pred = rfc_tuned.predict(x_test)
print(f"Tuned Model Accuracy: {accuracy_score(y_test, tuned_pred):.4f}")

print("\nClassification Report for Tuned Model:")
print(classification_report(y_test, tuned_pred, target_names=df['class'].unique()))

# --- 8. Confusion Matrix ---
print("\nVisualizing confusion matrix for tuned model...")
cm = confusion_matrix(y_test, tuned_pred)
class_names = df['class'].unique()

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for Tuned Model', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.savefig('github projects/car_eval_random_forest/confusion_matrix.png', bbox_inches='tight')
plt.show()

# --- 9. Feature Importance ---
print("\nAnalyzing feature importances...")
feature_scores = pd.Series(rfc_tuned.feature_importances_, index=x_train.columns).sort_values(ascending=False)
print("Feature Importances:")
print(feature_scores)

plt.figure(figsize=(12, 7))
sns.barplot(x=feature_scores, y=feature_scores.index, palette='mako')
plt.title('Feature Importance', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.savefig('github projects/car_eval_random_forest/feature_importance.png', bbox_inches='tight')
plt.show()

# --- 10. Analysis with Reduced Features ---
print("\nAnalyzing performance with a reduced feature set...")
X_reduced = df_encoded.drop(['class', 'doors', 'lug_boot'], axis=1)
x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(X_reduced, y, test_size=0.3, random_state=42, stratify=y)

rfc_reduced = RandomForestClassifier(random_state=42)
rfc_reduced.fit(x_train_r, y_train_r)
reduced_pred = rfc_reduced.predict(x_test_r)
print(f"Accuracy with reduced features: {accuracy_score(y_test_r, reduced_pred):.4f}")
print("\nScript finished.")