import joblib
import pandas as pd

model = joblib.load('model/category_model.pkl')

print("Model loaded successfully.")
print('Type "exit" to quit the testing program.')

while True:
    title = input(" Enter the product title: ")
    if title.lower() == "exit":
        print("Exiting...")
        break
# Predict category
# Create DataFrame with the same structure as during training
    X_input = pd.DataFrame(
        {"Product Title": [title]}
    )

    prediction = model.predict(X_input)
    print(f" Predicted Category: {prediction[0]}")
