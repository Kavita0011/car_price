import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import re # Import Regular Expressions

# 1. LOAD DATA
df = pd.read_csv('car details v3.csv')

# --------------------------------------------------------------------------
# THE "BULLETPROOF" CLEANING SECTION
# --------------------------------------------------------------------------

# This function removes ANY non-numeric characters (except the decimal point)
def clean_numeric(value):
    if pd.isna(value):
        return np.nan
    # This regex finds only the numbers and the dot: 12.3, 100, 45.67 etc.
    match = re.search(r"(\d+\.?\d*)", str(value))
    return float(match.group(1)) if match else np.nan

print("Cleaning units... please wait...")
# Apply the bulletproof cleaner to the three problem columns
df['mileage'] = df['mileage'].apply(clean_numeric)
df['engine'] = df['engine'].apply(clean_numeric)
df['max_power'] = df['max_power'].apply(clean_numeric)

# Now map the categorical text to numbers
df['fuel'] = df['fuel'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4})
df['transmission'] = df['transmission'].map({'Manual': 0, 'Automatic': 1})

# IMPORTANT: Fill NaNs now. 
# After cleaning units, some rows might become NaN. We MUST fill them.
df = df.fillna(df.median(numeric_only=True))
# --------------------------------------------------------------------------

# 2. VISUALIZATION (Optional - kept short for speed)
# list of columns tovisualize columns vs selling price
columns=['year', 'km_driven', 'fuel',  'mileage', 'engine', 'max_power']
# for col in columns:
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x=df[col], y=df['selling_price'])
#     plt.title(f'Scatter plot of {col} vs Selling Price')
#     plt.xlabel(col)
#     plt.ylabel('Selling Price')
#     plt.show()

# detect outliers and visualize distributions
# plt.figure(figsize=(10, 6))
# sns.histplot(df['selling_price'], kde=True)


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 3. OUTLIER REMOVAL
columns_to_check = ['year','selling_price', 'km_driven', 'mileage', 'engine', 'max_power']
for col in columns_to_check:
    df = df[df[col] < df[col].quantile(0.95)]  # Remove top 5% outliers

# 4. MODELING
# Define X and y
# getg current year and calculate car age
from datetime import datetime
year=current_year = datetime.now().year


df['car_age'] = year - df['year']
# Then change your X to:
X = df[['car_age', 'km_driven', 'fuel', 'mileage', 'engine', 'max_power']]
y = df['selling_price']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
# model = LinearRegression()
model=RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
print(f"\n✅ SUCCESS! R-squared value: {r2_score(y_test, y_pred):.4f}")
