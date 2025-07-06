# main.py
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import os

app = FastAPI(title="Music Recommender MLOps API", version="1.0.0")

# Set URI for MLflow tracing server (Optional, Default: './mlruns')
# os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# Variable to store the loaded model
model = None
MODEL_URI = None # Eg: "runs:/<YOUR_RUN_ID>/simple_recommender"

class RecommendRequest(BaseModel):
    user_id: str # User ID (Not used in this example, but required for actual system)
    num_recommendations: int = 5 # Songs to be recommended

@app.on_event("startup")
async def load_model():
    """
    Load model from MLflow when application starts
    """
    global model, MODEL_URI

    # You can latest Run ID, Or specify target Run ID
    # Here, it shows the way of loading latest logged model
    # For actual production env, it is better to specify target model version
    try:
        # Find latest run using MLflow client
        # Caution: this way may be less robust than specifying target run_id
        # For actual deployment, Use clear MODEL_URI (Eg: "models:/model_name/Production")
        runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
        if not runs.empty:
            latest_run_id = runs.iloc[0].run_id
            MODEL_URI = f"runs:/{latest_run_id}/simple_recommender"
            print(f"Attempting to load model from: {MODEL_URI}")
            model = mlflow.pyfunc.load_model(MODEL_URI)
            print("Model loaded successfully!")
        else:
            print("No MLflow runs found. Please run recommender_model.py first.")
            model = None
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        model = None

@app.get("/")
async def root():
    return {"message": "Welcome to the Music Recommender API! Use /recommend to get recommendations."}

@app.post("/recommend")
async def recommend_music(request: RecommendRequest):
    """
    Recommend musics to users
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Recommendation model not loaded.")

    try:
        # Model's predict method gets context and model_input
        # This example model does not use model_input, therefore it delivers empty DataFrame
        # In actual model, personalized recommendation using request.user_id or and so on.
        recommendations = model.predict(None, pd.DataFrame([{"user_id": request.user_id}]))
        return {"user_id": request.user_id, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during recommendation: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

