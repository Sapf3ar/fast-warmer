import argparse
import pandas as pd
import numpy as np
import faiss
import json
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class VideoStatNeighborFinder:
    """Class to find and manage video neighbors based on statistical features."""

    def __init__(self, parquet_path, numerical_features, categorical_features, log_features, k, index_path):
        """
        Initializes the VideoStatNeighborFinder with the specified parameters.

        Args:
            parquet_path (str): Path to the parquet file containing video data.
            numerical_features (list): List of numerical feature names.
            categorical_features (list): List of categorical feature names.
            log_features (list): List of features to apply log1p transformation.
            k (int): Number of nearest neighbors to search for.
            index_path (str): Path to save or load the FAISS index.
        """
        self.parquet_path = parquet_path
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.log_features = log_features
        self.k = k
        self.index_path = index_path
        self.df = None
        self.feature_matrix = None
        self.index = None
        self.neighbors_dict = None
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    def load_data(self):
        """Loads data from the parquet file."""
        self.df = pd.read_parquet(self.parquet_path)

    def preprocess_features(self):
        """Preprocesses numerical and categorical features."""
        # Apply log1p transformation to specified features
        self.df[self.log_features] = self.df[self.log_features].applymap(np.log1p)

        # Combine numerical and log-transformed features
        all_numerical_features = self.numerical_features + self.log_features

        # Scale numerical features
        numerical_data = self.scaler.fit_transform(self.df[all_numerical_features])

        # Encode categorical features
        categorical_data = self.encoder.fit_transform(self.df[self.categorical_features])

        # Combine all features into a single feature matrix
        self.feature_matrix = np.hstack((numerical_data, categorical_data)).astype('float32')

    def build_index(self):
        """Builds a FAISS index from the feature matrix."""
        self.index = faiss.IndexFlatL2(self.feature_matrix.shape[1])
        self.index.add(self.feature_matrix)
        faiss.write_index(self.index, self.index_path)

    def load_index(self):
        """Loads a FAISS index from the specified path."""
        self.index = faiss.read_index(self.index_path)

    def search_neighbors(self):
        """Searches for k nearest neighbors in the index."""
        distances, indices = self.index.search(self.feature_matrix, self.k)
        video_ids = self.df['video_id'].values
        self.neighbors_dict = {}
        for i, video_id in enumerate(video_ids):
            neighbor_ids = video_ids[indices[i]].tolist()
            self.neighbors_dict[video_id] = neighbor_ids

    def save_results(self):
        """Saves the neighbors dictionary to JSON and CSV files."""
        # Save to JSON
        with open('stat_neighbors.json', 'w') as json_file:
            json.dump(self.neighbors_dict, json_file, indent=4)

        # Prepare data for CSV
        records = []
        for video_id, neighbor_ids in self.neighbors_dict.items():
            for neighbor_id in neighbor_ids:
                records.append({
                    'video_id': video_id,
                    'neighbor_video_id': neighbor_id
                })
        neighbors_df = pd.DataFrame(records)
        neighbors_df.to_csv("stat_neighbors.csv", index=False)

    def add_video(self, new_video_data):
        """
        Adds a new video to the index and updates neighbors.

        Args:
            new_video_data (dict): Dictionary containing the new video's data.
        """
        new_video_df = pd.DataFrame([new_video_data])

        # Apply log1p transformation to specified features
        new_video_df[self.log_features] = new_video_df[self.log_features].applymap(np.log1p)

        # Combine numerical and log-transformed features
        all_numerical_features = self.numerical_features + self.log_features

        # Scale numerical features using existing scaler
        numerical_data = self.scaler.transform(new_video_df[all_numerical_features])

        # Encode categorical features using existing encoder
        categorical_data = self.encoder.transform(new_video_df[self.categorical_features])

        # Combine all features into a single feature vector
        new_feature_vector = np.hstack((numerical_data, categorical_data)).astype('float32')

        # Add new data to existing dataframes
        self.df = pd.concat([self.df, new_video_df], ignore_index=True)
        self.feature_matrix = np.vstack((self.feature_matrix, new_feature_vector))

        # Add new vector to the index
        self.index.add(new_feature_vector)

        # Update the index file
        faiss.write_index(self.index, self.index_path)

        # Re-search neighbors with the updated index
        self.search_neighbors()
        self.save_results()

def main():
    parser = argparse.ArgumentParser(description='Find video neighbors using statistical features.')
    parser.add_argument('--parquet_path', type=str, required=True, help='Path to the parquet file.')
    parser.add_argument('--k', type=int, default=300, help='Number of nearest neighbors to search for.')
    parser.add_argument('--index_path', type=str, default='stat_video_index.faiss', help='Path to save/load the FAISS index.')
    args = parser.parse_args()

    numerical_features = [
        'v_total_comments', 'v_likes', 'v_dislikes', 'v_duration',
        'v_cr_click_like_7_days', 'v_cr_click_dislike_7_days', 'v_cr_click_vtop_7_days',
        'v_cr_click_long_view_7_days', 'v_cr_click_comment_7_days', 'v_cr_click_like_30_days',
        'v_cr_click_dislike_30_days', 'v_cr_click_vtop_30_days', 'v_cr_click_long_view_30_days',
        'v_cr_click_comment_30_days', 'v_cr_click_like_1_days', 'v_cr_click_dislike_1_days',
        'v_cr_click_vtop_1_days', 'v_cr_click_long_view_1_days', 'v_cr_click_comment_1_days',
        'v_avg_watchtime_1_day', 'v_avg_watchtime_7_day', 'v_avg_watchtime_30_day',
        'v_frac_avg_watchtime_1_day_duration', 'v_frac_avg_watchtime_7_day_duration',
        'v_frac_avg_watchtime_30_day_duration', 'v_category_popularity_percent_7_days',
        'v_category_popularity_percent_30_days', 'v_long_views_1_days', 'v_long_views_7_days',
        'v_long_views_30_days'
    ]

    log_features = ['v_year_views', 'v_month_views', 'v_week_views', 'v_day_views']

    categorical_features = ['category_id']

    finder = VideoStatNeighborFinder(
        parquet_path=args.parquet_path,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        log_features=log_features,
        k=args.k,
        index_path=args.index_path
    )

    finder.load_data()
    finder.preprocess_features()
    finder.build_index()
    finder.search_neighbors()
    finder.save_results()

if __name__ == "__main__":
    main()
