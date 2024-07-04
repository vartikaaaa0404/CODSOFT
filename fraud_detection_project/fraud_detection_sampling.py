import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, auc, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Load the dataset
df = pd.read_csv('/Users/vartikarawat/Desktop/internship/fraud_detection_project/creditcard.csv')

# Sample a fraction of the dataset for analysis
df_sample = df.sample(frac=0.1, random_state=42)

# Exploratory Data Analysis (EDA)
print(df_sample.head())
print(df_sample.info())
print(df_sample.describe())
print(df_sample['Class'].value_counts())

# Handling missing values
if df_sample.isnull().sum().any():
    df_sample = df_sample.dropna()

# Normalizing amount and time features
scaler = StandardScaler()
df_sample['Normalized_Amount'] = scaler.fit_transform(df_sample['Amount'].values.reshape(-1, 1))
df_sample['Normalized_Time'] = scaler.fit_transform(df_sample['Time'].values.reshape(-1, 1))
df_sample.drop(['Amount', 'Time'], axis=1, inplace=True)

# Splitting the data
X = df_sample.drop('Class', axis=1)
y = df_sample['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balancing classes
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Define base models
base_models = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('lr', LogisticRegression(random_state=42, max_iter=1000)),
    ('ada', AdaBoostClassifier(random_state=42, algorithm='SAMME')),
    ('mlp', MLPClassifier(random_state=42))
]

# Define stacking classifier
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    passthrough=True
)

# Training the stacking model
stacking_model.fit(X_train_res, y_train_res)

# Making predictions
y_pred = stacking_model.predict(X_test)

# Classification Report
print("Classification Report for Stacking Model:")
print(classification_report(y_test, y_pred))

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Precision-Recall Curve and AUPRC
precision, recall, _ = precision_recall_curve(y_test, y_pred)
auprc = auc(recall, precision)
print(f'AUPRC: {auprc}')

# Model interpretation
result = permutation_importance(stacking_model, X_test, y_test, n_repeats=10, random_state=42)
feature_importances = pd.Series(result.importances_mean, index=X_test.columns)

# Plot the top 10 feature importances
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
top_features = feature_importances.nlargest(10)
top_features.plot(kind='barh')
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances')
plt.tight_layout()

# Save the plot to a file
if not os.path.exists('plots'):
    os.makedirs('plots')
plt.savefig('plots/feature_importances.png')


plt.subplot(1, 2, 2)
plt.plot(recall, precision, label='Stacking Classifier')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.tight_layout()

# Save the plot to a file
plt.savefig('plots/precision_recall_curve.png')


# Deep Learning Model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_res.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_res, y_train_res, epochs=10, batch_size=32, validation_split=0.2)

# Real-time Detection
y_pred_dl = model.predict(X_test)
y_pred_dl = (y_pred_dl > 0.5).astype(int)

# Classification Report for Deep Learning Model
print("Classification Report for Deep Learning Model:")
print(classification_report(y_test, y_pred_dl))

# Accuracy Score for Deep Learning Model
accuracy_dl = accuracy_score(y_test, y_pred_dl)
print(f'Accuracy for Deep Learning Model: {accuracy_dl}')

print("The code has completed successfully.")
