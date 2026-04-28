from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = FastAPI()

# Train model at startup
iris = load_iris()
model = DecisionTreeClassifier(random_state=42)
model.fit(iris.data, iris.target)
class_names = ["setosa", "versicolor", "virginica"]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/predict")
async def predict(sl: float, sw: float, pl: float, pw: float):
    # Format the incoming query parameters into a 2D array for the model
    features = np.array([[sl, sw, pl, pw]])
    
    # Predict the class index (0, 1, or 2)
    # Cast to int() to avoid JSON serialization errors with numpy int64
    pred = int(model.predict(features)[0])
    
    return {
        "prediction": pred, 
        "class_name": class_names[pred]
    }
