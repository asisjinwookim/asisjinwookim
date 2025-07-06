
# Install Conda
- https://repo.anaconda.com/archive/

# Create new conda environment
conda create -n music_recommender python=3.9

# Activate env
conda activate music_recommender

# Install required packages
pip install fastapi uvicorn pandas scikit-learn mlflow


# Train model & log to MLFlow
python recommender_model.py

# Check Conda env is activated: conda activate music_recommender
uvicorn main:app --reload --host 0.0.0.0 --port 8000
