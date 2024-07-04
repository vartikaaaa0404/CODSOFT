import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D

# Load the Iris dataset from CSV
data = pd.read_csv('IRIS.csv')

# Data description
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset description:")
print(data.describe())

print("\nSpecies value counts:")
print(data['species'].value_counts())

# Data visualization - Pie chart
species_counts = data['species'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Iris Species")
plt.savefig("pie_chart.png")
plt.close()

# Data visualization - Scatter plots
sns.pairplot(data, hue='species', markers=["o", "s", "D"])
plt.savefig("pairplot.png")
plt.close()

# Data visualization - 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=data['species'].factorize()[0], cmap='viridis')
ax.set_xlabel(data.columns[0])
ax.set_ylabel(data.columns[1])
ax.set_zlabel(data.columns[2])
plt.title("3D Scatter Plot of Iris Dataset")
plt.savefig("3d_scatter_plot.png")
plt.close()

# Preprocess the data
X = data.drop(columns=['species'])
y = data['species'].factorize()[0]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)

# Train a Random Forest model with hyperparameter tuning
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train_standard, y_train)

print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
print("Best Cross-validation Accuracy for Random Forest:", grid_search_rf.best_score_)

# Use best model from Grid Search
best_rf = grid_search_rf.best_estimator_

# Evaluate on test set
y_pred_best_rf = best_rf.predict(X_test_standard)
print("\nBest Random Forest Classification Report:")
print(classification_report(y_test, y_pred_best_rf))
print("Best Random Forest Accuracy:", accuracy_score(y_test, y_pred_best_rf))

# Error analysis - Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_best_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=data['species'].unique(), yticklabels=data['species'].unique())
plt.title('Best Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("best_rf_confusion_matrix.png")
plt.close()

# Feature importance analysis
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.savefig("feature_importance.png")
plt.close()

# Train an Artificial Neural Network (ANN) model using MLPClassifier
param_grid_ann = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [1000, 2000, 3000],  # Adjusted max_iter values
    'tol': [1e-4, 1e-3]
}

ann = MLPClassifier(random_state=42)
grid_search_ann = GridSearchCV(estimator=ann, param_grid=param_grid_ann, cv=5, scoring='accuracy')
grid_search_ann.fit(X_train_standard, y_train)

print("Best Parameters for ANN:", grid_search_ann.best_params_)
print("Best Cross-validation Accuracy for ANN:", grid_search_ann.best_score_)

# Use best model from Grid Search
best_ann = grid_search_ann.best_estimator_

# Evaluate on test set
y_pred_best_ann = best_ann.predict(X_test_standard)
print("\nBest ANN Classification Report:")
print(classification_report(y_test, y_pred_best_ann))
print("Best ANN Accuracy:", accuracy_score(y_test, y_pred_best_ann))

# Error analysis - Confusion Matrix
conf_matrix_ann = confusion_matrix(y_test, y_pred_best_ann)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_ann, annot=True, fmt='d', cmap='Blues', xticklabels=data['species'].unique(), yticklabels=data['species'].unique())
plt.title('Best ANN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("best_ann_confusion_matrix.png")
plt.close()

print("Code completed.")
