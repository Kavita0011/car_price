# ---------------------------------- carsdata linear regression ------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 

# ---------------------------------- loading dataset ------------------------------------------
df = pd.read_csv('car data.csv')

# --------------------------------- initial inspection
print(df.head())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
 
# FIX: Use exact column names from the CSV
df['Fuel_Type'] = df['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
df['Transmission'] = df['Transmission'].map({'Manual': 0, 'Automatic': 1})
df['Seller_Type'] = df['Seller_Type'].map({'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2})

# --------------------------------- data visualization
graph_list = ['Year', 'Kms_Driven', 'Fuel_Type', 'Transmission', 'Selling_Price']

for col in graph_list:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# see correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# --------------------------------- Outlier Removal ---------------------------------
# Removing top 5% of outliers for Selling Price and Kms Driven
df = df[(df['Selling_Price'] < df['Selling_Price'].quantile(0.95)) &
        (df['Kms_Driven'] < df['Kms_Driven'].quantile(0.95))]

# --------------------------------- Model Training ----------------------------------

# I added 'Present_Price' here because it's the most important feature for car prices!
X = df[['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Transmission','Seller_Type']]
y = df['Selling_Price']

# FIXED: The order must be X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Final R-squared score: {r2:.4f}")

# Visualizing Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()
