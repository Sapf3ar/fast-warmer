import pandas as pd
import lightgbm as lgb
from typing import List
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class Item(BaseModel):
    """
    Pydantic model representing a video item.
    """
    name: str
    description: str | None
    category: str | None
    video_id: str
    interaction: int = 0

class Ranker:
    """
    Ranker class that trains a LightGBM model to rank videos based on user interactions.
    """
    def __init__(self, 
                 logs_path: str,
                 stats_path: str,
                 model_params: dict,
                 categorical_columns: list,
                 target_column: str = 'target',
                 model_path: str = None):
        """
        Initializes the Ranker class.

        Args:
            logs_path (str): Path to the logs data file.
            stats_path (str): Path to the video statistics data file.
            model_params (dict): Parameters for the LightGBM model.
            categorical_columns (list): List of categorical feature columns.
            target_column (str): Name of the target variable.
            model_path (str): Path to save/load the model.
        """
        self.logs_path = logs_path
        self.stats_path = stats_path
        self.model_params = model_params
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self.model_path = model_path
        self.model = None
        self.label_encoders = {}
        self.features = None
        self.target = None
        self.data = None

    def load_data(self):
        """Loads and merges the logs and stats data."""
        data = pd.read_parquet(self.logs_path)
        stats = pd.read_parquet(self.stats_path)
        self.data = pd.merge(data, stats, on='video_id', how='left')

    def preprocess_data(self):
        """Preprocesses the data for training."""
        # Replace zero durations to avoid division by zero
        self.data['v_duration'] = self.data['v_duration'].replace(0, 1e-6)

        # Calculate target variable
        self.data[self.target_column] = self.data['watchtime'] / self.data['v_duration']
        self.data[self.target_column] = self.data[self.target_column].apply(lambda x: min(x, 1))

        # Label encode categorical columns
        for column in self.categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            self.data[column] = self.label_encoders[column].fit_transform(self.data[column].astype(str))

        # Drop rows with NaN values
        self.data = self.data.dropna()

        # Define features and target
        self.features = self.data.drop(columns=[
            'event_timestamp', 'user_id', 'video_id', 'watchtime', 
            'v_pub_datetime', 'title', 'description', self.target_column
        ])
        self.target = self.data[self.target_column]

    def train_model(self):
        """Trains the LightGBM model."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        self.model = lgb.LGBMRegressor(**self.model_params)
        self.model.fit(
            X_train, 
            y_train, 
            eval_set=[(X_test, y_test)],
            eval_metric='rmse',
            categorical_feature=self.categorical_columns,
            verbose=False
        )
        # Evaluate model
        y_pred = self.model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        print(f"RMSE on test set: {rmse:.4f}")
        print(f"R² on test set: {r2:.4f}")

    def save_model(self):
        """Saves the trained model to disk."""
        if self.model_path:
            joblib.dump({
                'model': self.model,
                'label_encoders': self.label_encoders
            }, self.model_path)
        else:
            raise ValueError("Model path not specified.")

    def load_model(self):
        """Loads the model from disk."""
        if self.model_path:
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.label_encoders = data['label_encoders']
        else:
            raise ValueError("Model path not specified.")

    def preprocess_candidates(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the candidate data for prediction.

        Args:
            candidates_df (pd.DataFrame): DataFrame containing candidate videos.

        Returns:
            pd.DataFrame: Preprocessed features for prediction.
        """
        # Fill missing values
        for column in self.categorical_columns:
            candidates_df[column] = candidates_df[column].astype(str).fillna('Unknown')

        for column in candidates_df.columns:
            if candidates_df[column].dtype in ['int64', 'float64']:
                candidates_df[column] = candidates_df[column].fillna(candidates_df[column].median())
            elif candidates_df[column].dtype == 'object':
                candidates_df[column] = candidates_df[column].fillna('Unknown')

        # Label encode categorical columns
        for column in self.categorical_columns:
            candidates_df[column] = self.label_encoders[column].transform(candidates_df[column].astype(str))

        # Define features
        X_candidates = candidates_df.drop(columns=['user_id', 'video_id'])
        return X_candidates

    def predict(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Predicts scores for the candidate videos.

        Args:
            candidates_df (pd.DataFrame): DataFrame containing candidate videos.

        Returns:
            pd.DataFrame: Candidates DataFrame with predicted scores.
        """
        X_candidates = self.preprocess_candidates(candidates_df)
        candidates_df['score'] = self.model.predict(X_candidates)
        return candidates_df

    def generate(self, user_id: int, candidates_df: pd.DataFrame, top_k: int = 10) -> List[Item]:
        """Generates ranked recommendations for the user.

        Args:
            user_id (int): The user ID for which to generate recommendations.
            candidates_df (pd.DataFrame): DataFrame containing candidate videos.
            top_k (int): Number of top recommendations to return.

        Returns:
            List[Item]: List of top K recommended items.
        """
        user_candidates = candidates_df[candidates_df['user_id'] == user_id].copy()
        if user_candidates.empty:
            return []
        user_candidates = self.predict(user_candidates)
        user_candidates_sorted = user_candidates.sort_values(by='score', ascending=False).head(top_k)
        result_items = [
            Item(
                name=row['title'],
                description=row['description'],
                category=str(row['category_id']),
                video_id=row['video_id'],
                interaction=0
            )
            for _, row in user_candidates_sorted.iterrows()
        ]
        return result_items
