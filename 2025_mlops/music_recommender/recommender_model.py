# recommender_model.py
import pandas as pd
import random
import mlflow
import mlflow.pyfunc
from typing import Dict, List, Any
import mlflow.models # Make sure this import is present

class SimpleMusicRecommender(mlflow.pyfunc.PythonModel):
    def __init__(self, top_n: int = 5):
        """
        Initializes the music recommender with a specified number of top recommendations.

        Args:
            top_n (int): The number of top popular songs to recommend.
        """
        self.top_n = top_n
        self.music_data = self._generate_dummy_data()

    def _generate_dummy_data(self) -> pd.DataFrame:
        """
        Generates dummy music data for demonstration purposes.
        In a real application, this would load data from a database or a file.

        Returns:
            pd.DataFrame: A DataFrame containing dummy music information.
        """
        data = {
            'music_id': [f'm{i:03d}' for i in range(1, 21)],
            'title': [f'Song Title {i}' for i in range(1, 21)],
            'artist': [f'Artist {i%5 + 1}' for i in range(1, 21)],
            'popularity': [random.randint(1, 100) for _ in range(20)]
        }
        return pd.DataFrame(data)

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Recommends the top N popular songs from the dummy dataset.
        Although `model_input` is provided as per MLflow's pyfunc signature,
        it is not used in this simple example for personalized recommendations.
        It can be used to pass user IDs or other features for custom recommendations.

        Args:
            context (mlflow.pyfunc.PythonModelContext): MLflow context object.
            model_input (pd.DataFrame): Input data for prediction (e.g., user IDs).
                                        Not used in this simplified model.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a recommended song.
        """
        recommended_music = self.music_data.sort_values(by='popularity', ascending=False).head(self.top_n)
        return recommended_music[['music_id', 'title', 'artist']].to_dict(orient='records')

def train_and_log_model():
    """
    Instantiates and logs the SimpleMusicRecommender model to MLflow.
    """
    with mlflow.start_run(run_name="simple_music_recommender_training"):
        recommender = SimpleMusicRecommender(top_n=5)

        # Define an example input for the model.
        example_input = pd.DataFrame([{"user_id": "example_user_001"}])

        # Get an example output from the model using the example input.
        example_output = recommender.predict(None, example_input)

        # Get the current MLflow version to pin it in conda_env
        import pkg_resources
        mlflow_version = pkg_resources.get_distribution("mlflow").version
        print(f"Logging model with MLflow version: {mlflow_version}")

        mlflow.pyfunc.log_model(
            name="simple_recommender",
            python_model=recommender,
            input_example=example_input,
            signature=mlflow.models.infer_signature(example_input, example_output),
            conda_env={
                "channels": ["defaults"],
                "dependencies": [
                    "python=3.9",
                    "pandas",
                    "scikit-learn",
                    f"mlflow=={mlflow_version}"  # Pin the exact MLflow version
                ],
            }
        )
        mlflow.log_param("top_n", recommender.top_n)
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"Model logged at: runs:/{mlflow.active_run().info.run_id}/artifacts/simple_recommender")

if __name__ == "__main__":
    # To view the MLflow UI locally, run 'mlflow ui' in your terminal.
    # You can set the MLflow tracking server URI:
    # export MLFLOW_TRACKING_URI="http://localhost:5000" (default)
    train_and_log_model()

