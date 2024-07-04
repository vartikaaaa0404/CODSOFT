import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats.mstats import winsorize

# Load the dataset
df = pd.read_csv('/Users/vartikarawat/Desktop/internship/sales_prediction/advertising (1).csv')

# Data cleaning
# Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# Check for duplicates
print("Duplicates:")
print(df.duplicated().sum())

# Outlier analysis
plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
sns.boxplot(df['TV'])
plt.title('TV Advertisement')
plt.subplot(1, 3, 2)
sns.boxplot(df['Radio'])
plt.title('Radio Advertisement')
plt.subplot(1, 3, 3)
sns.boxplot(df['Newspaper'])
plt.title('Newspaper Advertisement')
plt.savefig('outlier_analysis.png')

# Handling outliers (Winsorization)
df['TV'] = winsorize(df['TV'], limits=[0.05, 0.05])
df['Radio'] = winsorize(df['Radio'], limits=[0.05, 0.05])
df['Newspaper'] = winsorize(df['Newspaper'], limits=[0.05, 0.05])

# EDA
plt.figure(figsize=(10, 6))
sns.histplot(df['Sales'], bins=20, kde=True)
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.title('Sales Distribution')
plt.savefig('sales_distribution.png')

# Sales relation with other variables
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
sns.scatterplot(data=df, x='TV', y='Sales')
plt.title('TV Advertisement vs Sales')
plt.subplot(1, 3, 2)
sns.scatterplot(data=df, x='Radio', y='Sales')
plt.title('Radio Advertisement vs Sales')
plt.subplot(1, 3, 3)
sns.scatterplot(data=df, x='Newspaper', y='Sales')
plt.title('Newspaper Advertisement vs Sales')
plt.tight_layout()
plt.savefig('sales_vs_advertisements.png')

# Heatmap for correlation
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')

# Additional Box Plots and Sales Relation Plots
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.boxplot(data=df, x='TV')
plt.title('Box Plot for TV Advertisement')
plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='Radio')
plt.title('Box Plot for Radio Advertisement')
plt.subplot(2, 2, 3)
sns.boxplot(data=df, x='Newspaper')
plt.title('Box Plot for Newspaper Advertisement')
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='Sales', y='TV')
plt.title('Sales vs. TV Advertisement')
plt.tight_layout()
plt.savefig('additional_box_scatter_plots.png')

# Split the data into features (X) and target (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline to scale features and fit a linear regression model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', LinearRegression())
])

# Cross-validation to evaluate the model
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Cross-validated R-squared scores: {cv_scores}")
print(f"Mean CV R-squared: {np.mean(cv_scores)}")

# Fit the model on the entire dataset
model.fit(X, y)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualizing the fit on the test set
plt.figure(figsize=(10, 6))
plt.scatter(X_test['TV'], y_test, label='Actual')
plt.scatter(X_test['TV'], y_pred, color='red', label='Predicted')
plt.xlabel('TV Advertisement')
plt.ylabel('Sales')
plt.title('Linear Regression Fit on Test Set (TV vs. Sales)')
plt.legend()
plt.savefig('regression_fit_tv_vs_sales.png')

# Print metrics
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

print("The code has completed successfully.")
