import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

print("Starting model training process...")

# 1. Load Data
try:
    # --- THIS IS THE MODIFIED SECTION ---
    # We now load the local file 'titanic.csv'
    df = pd.read_csv('titanic.csv')
    print("Titanic dataset loaded successfully from 'titanic.csv'.")
    # ------------------------------------
except FileNotFoundError:
    print("Error: 'titanic.csv' file not found.")
    print("Please make sure 'titanic.csv' is in the same folder as this script.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 2. Define Features (X) and Target (y)
# The column names are the same as the seaborn dataset
features = ['pclass', 'sex', 'age', 'fare', 'embarked']
target = 'survived'

# Rename columns to lowercase to match the new file if needed
# (The uploaded file already has lowercase headers, but this is good practice)
df.columns = df.columns.str.lower()

# Drop rows where 'survived' or 'embarked' is missing
df = df[features + [target]].dropna(subset=[target, 'embarked'])
df['pclass'] = df['pclass'].astype(str) # Treat pclass as categorical

print("Data pre-processing started...")

# 3. Data Preprocessing & Feature Engineering Pipeline
numeric_features = ['age', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['pclass', 'sex', 'embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Create the Full Model Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Split Data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the model...")

# 6. Train the Model
model.fit(X_train, y_train)

# 7. Evaluate Model
accuracy = model.score(X_test, y_test)
print(f"Model trained successfully. Accuracy: {accuracy:.4f}")

# 8. Serialize and Save the Model
joblib.dump(model, 'titanic_model.pkl')
print("Model serialized and saved as 'titanic_model.pkl'")