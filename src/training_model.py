import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

df = pd.read_csv('products.csv') # Load your dataset

#Standardizing column names by removing leading/trailing spaces and underscore on the beginning
df.columns = df.columns.str.strip().str.lstrip('_')

# # Deleting unnecessary columns
columns_to_delete = ['Number_of_Views', 'Merchant Rating', 'Listing Date', 'Product Code']
df.drop(columns=columns_to_delete, inplace=True)

# Deleting rows with missing values in important columns
important_columns = ['Product Title', 'Category Label']
df.dropna(subset=important_columns, inplace=True)

# Resetting the index after row deletions
df.reset_index(drop=True, inplace=True)

# Standardize names in category column
df['Category Label'] = df['Category Label'].replace({'Fridges':'Fridge Freezers', 
                        'Freezers':'Fridge Freezers', 
                        'fridge':'Fridge Freezers',
                        'Mobile Phone': 'Mobile Phones',
                        'CPUs': 'CPU',})

# Splitting the dataset into features and target variable
X = df[['Product Title']]
y = df['Category Label']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "Product Title")
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LinearSVC())
])

# Train the model on the entire dataset
pipeline.fit(X, y)

# Save the trained model
joblib.dump(pipeline, 'model/category_model.pkl')

print("Model training completed and saved as 'model/category_model.pkl'")