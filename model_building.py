import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load the dataset
df1 = pd.read_csv('winequality-white.csv',sep=";")  # Replace with your dataset
# Data Cleaning
df1.dropna(inplace=True)  # Removing missing data

df1['wine_class'] = df1['quality'].apply(lambda x: 1 if x >= 7 else (0))

# Data Preprocessing
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'total sulfur dioxide',
             'pH',  'alcohol']
X = df1[features]
y = df1['wine_class']


# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
scaled_X = scaler_x.fit_transform(X_train)


y_train_reshaped = y_train.values.reshape(-1, 1)  # If y_train is a Pandas Series
scaler_y = StandardScaler()
scaled_y = scaler_y.fit_transform(y_train_reshaped)  # Now this will work



rf_classifier = RandomForestClassifier(max_depth=5, min_samples_split=5, random_state=33)

rf_classifier.fit(scaled_X, y_train)

# Step 6: Save the scaler and model
joblib.dump(scaler_x, 'scaler_x.joblib')
joblib.dump(scaler_y, 'scaler_y.joblib')
joblib.dump(rf_classifier, 'random_forest_model.joblib')


