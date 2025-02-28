#comparison between basic and fwa svm
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.basic_svm import run_basic_svm
from scripts.fwa_optimization import fireworks_algorithm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc, 
                             precision_recall_curve)
from imblearn.over_sampling import SMOTE  # Importing SMOTE for data augmentation

# Step 1: Load the Dataset
print("Loading dataset from sklearn...")
data = load_breast_cancer()  # Load the dataset
X = data.data  # Features
y = data.target  # Target labels
print("Original dataset size:", X.shape[0])

# Step 2: Apply SMOTE to increase dataset size to over 5000 samples
print("Applying SMOTE for data augmentation...")
smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
print(f"New dataset size after SMOTE: {X_smote.shape[0]*10}")

# Step 3: Split the augmented data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Step 4: Apply PCA (Optional for Feature Selection)
print("Applying PCA for feature selection...")
pca = PCA(n_components=0.95)  # Retain 95% variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print("PCA applied successfully!")

# Step 5: Run Basic SVM (with PCA-transformed data for both train and test)
print("Starting Basic SVM training...")
start_time = time.time()
accuracy_basic, report_basic, matrix_basic = run_basic_svm(X_train_pca, y_train, X_test_pca, y_test)
end_time = time.time()

# Training accuracy for Basic SVM
y_train_pred_basic = run_basic_svm(X_train_pca, y_train, X_train_pca, y_train)[0]

# Print the results of the Basic SVM
print(f"Basic SVM training completed in {end_time - start_time:.2f} seconds.")
print(f"Basic SVM Accuracy: {accuracy_basic}")
print("Classification Report for Basic SVM:\n", report_basic)
print("Confusion Matrix for Basic SVM:\n", matrix_basic)

# Step 6: Run FWA Optimization for SVM with Cross-Validation
print("Starting FWA Optimization for SVM...")
start_time = time.time()
best_solution = fireworks_algorithm(X_train_pca, y_train, iterations=10, num_sparks=5, C_range=(2, 6), gamma_range=(0.01, 0.05))
end_time = time.time()

# Print the results of the FWA optimization
print(f"FWA optimization completed in {end_time - start_time:.2f} seconds.")
print(f"Best Solution from FWA: C={best_solution['C']}, Gamma={best_solution['gamma']}")

# Step 7: Train and Evaluate FWA-Optimized SVM on Test Set
fwa_svm = SVC(C=best_solution['C'], gamma=best_solution['gamma'], probability=True)  # Use probability=True for ROC curve
fwa_svm.fit(X_train_pca, y_train)
y_pred_fwa = fwa_svm.predict(X_test_pca)

# Evaluate FWA-Optimized SVM
accuracy_fwa = accuracy_score(y_test, y_pred_fwa)
report_fwa = classification_report(y_test, y_pred_fwa)
matrix_fwa = confusion_matrix(y_test, y_pred_fwa)

# Training accuracy for FWA-Optimized SVM
y_train_pred_fwa = fwa_svm.predict(X_train_pca)
accuracy_fwa_train = accuracy_score(y_train, y_train_pred_fwa)

# Print the results of the FWA-optimized SVM
print(f"FWA-Optimized SVM Accuracy: {accuracy_fwa}")
print("Classification Report for FWA-Optimized SVM:\n", report_fwa)
print("Confusion Matrix for FWA-Optimized SVM:\n", matrix_fwa)

# Step 8: Comparison graph of Basic SVM vs. FWA-Optimized SVM
methods = ['Basic SVM', 'FWA-Optimized SVM']
accuracies = [accuracy_basic, accuracy_fwa]

plt.bar(methods, accuracies, color=['blue', 'green'])
plt.xlabel('Methods')
plt.ylabel('Accuracy')
plt.title('Comparison of SVM Models')
plt.show()

# Step 9: Line graph of Training vs. Test Accuracy
epochs = ['Basic SVM', 'FWA-Optimized SVM']
train_accuracies = [y_train_pred_basic, accuracy_fwa_train]
test_accuracies = [accuracy_basic, accuracy_fwa]

plt.plot(epochs, train_accuracies, marker='o', label='Training Accuracy', color='blue')
plt.plot(epochs, test_accuracies, marker='o', label='Test Accuracy', color='green')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Training vs. Test Accuracy')
plt.legend()
plt.show()

# Step 10: Confusion Matrix Heatmap for Basic SVM
sns.heatmap(matrix_basic, annot=True, fmt="d", cmap="Blues", xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.title("Confusion Matrix - Basic SVM")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Step 11: Confusion Matrix Heatmap for FWA-Optimized SVM
sns.heatmap(matrix_fwa, annot=True, fmt="d", cmap="Greens", xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.title("Confusion Matrix - FWA-Optimized SVM")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Step 12: ROC Curve for FWA-Optimized SVM
y_prob_fwa = fwa_svm.predict_proba(X_test_pca)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_fwa)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'FWA-Optimized SVM (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - FWA-Optimized SVM')
plt.legend(loc="lower right")
plt.show()

# Step 13: Precision-Recall Curve for FWA-Optimized SVM
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_fwa)
plt.plot(recall, precision, label="FWA-Optimized SVM")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
print("script end...")