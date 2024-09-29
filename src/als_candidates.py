import argparse
import numpy as np
import polars as pl
from glob import glob
from tqdm import tqdm
from collections import Counter
from scipy.sparse import csr_matrix
from implicit.cpu.als import AlternatingLeastSquares
from sklearn.metrics import average_precision_score
from typing import Optional, List


class AuthorVideoALSRecommender:
    """Class to train an ALS model and recommend videos based on similar authors."""

    def __init__(
        self,
        video_stats_path: str,
        logs_path_pattern: str,
        min_video_watch_fraction: float,
        min_videos_per_author: int,
        min_user_interactions: int,
        factors: int,
        regularization: float,
        iterations: int,
        num_recommendations: int,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Initializes the recommender with specified parameters.

        Args:
            video_stats_path (str): Path to the video statistics Parquet file.
            logs_path_pattern (str): Glob pattern to the logs Parquet files.
            min_video_watch_fraction (float): Minimum watch fraction to filter videos.
            min_videos_per_author (int): Minimum number of videos per author to keep the author.
            min_user_interactions (int): Minimum number of interactions per user to keep the user.
            factors (int): Number of latent factors in the ALS model.
            regularization (float): Regularization parameter for ALS.
            iterations (int): Number of iterations to train the ALS model.
            num_recommendations (int): Number of videos to recommend.
            model_path (Optional[str]): Path to save or load the ALS model.
        """
        self.video_stats_path = video_stats_path
        self.logs_path_pattern = logs_path_pattern
        self.min_video_watch_fraction = min_video_watch_fraction
        self.min_videos_per_author = min_videos_per_author
        self.min_user_interactions = min_user_interactions
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.num_recommendations = num_recommendations
        self.model_path = model_path

        self.video_stats = None
        self.authors = None
        self.user_to_idx = {}
        self.author_to_idx = {}
        self.idx_to_author = {}
        self.user_author_matrix = None
        self.model = None

    def load_video_stats(self):
        """Loads and filters video statistics data."""
        self.video_stats = pl.read_parquet(self.video_stats_path)
        self.video_stats = self.video_stats.filter(
            pl.col("v_frac_avg_watchtime_30_day_duration") > self.min_video_watch_fraction
        )
        # Filter authors with enough videos
        author_video_counts = (
            self.video_stats
            .groupby("author_id")
            .agg([pl.col("video_id").count().alias("video_count")])
        )
        filtered_authors = author_video_counts.filter(
            pl.col("video_count") >= self.min_videos_per_author
        )
        self.authors = np.unique(filtered_authors["author_id"])

    def prepare_user_author_matrix(self):
        """Prepares the user-author interaction matrix."""
        user_interaction_counts = Counter()
        rows = []
        cols = []

        logs_paths = glob(self.logs_path_pattern)

        # Build the interaction data
        for log_path in tqdm(logs_paths, desc="Processing logs"):
            log = pl.read_parquet(log_path)

            log_with_authors = log.join(
                self.video_stats.select(["video_id", "author_id"]),
                on="video_id",
                how="inner"
            )

            log_filtered = log_with_authors.filter(
                pl.col("author_id").is_in(self.authors)
            )

            user_ids = log_filtered["user_id"].to_list()
            author_ids = log_filtered["author_id"].to_list()

            # Count user interactions
            user_interaction_counts.update(user_ids)

            # Collect interactions
            for user_id, author_id in zip(user_ids, author_ids):
                rows.append(user_id)
                cols.append(author_id)

        # Filter users with enough interactions
        filtered_users = {
            user_id for user_id, count in user_interaction_counts.items()
            if count >= self.min_user_interactions
        }
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(filtered_users))}
        self.author_to_idx = {author_id: idx for idx, author_id in enumerate(sorted(self.authors))}
        self.idx_to_author = {idx: author_id for author_id, idx in self.author_to_idx.items()}

        # Build indices for the matrix
        rows_idx = []
        cols_idx = []
        for user_id, author_id in zip(rows, cols):
            if user_id in self.user_to_idx and author_id in self.author_to_idx:
                rows_idx.append(self.user_to_idx[user_id])
                cols_idx.append(self.author_to_idx[author_id])

        data = np.ones(len(rows_idx), dtype=np.float32)
        self.user_author_matrix = csr_matrix(
            (data, (rows_idx, cols_idx)),
            shape=(len(self.user_to_idx), len(self.author_to_idx))
        )

    def train_model(self):
        """Trains the ALS model."""
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations
        )
        self.model.fit(self.user_author_matrix)

        if self.model_path:
            self.model.save(self.model_path)

    def load_model(self):
        """Loads a pre-trained ALS model."""
        if self.model_path:
            self.model = AlternatingLeastSquares()
            self.model.load(self.model_path)
        else:
            raise ValueError("Model path not provided.")

    def recommend_videos(self, author_id: int) -> Optional[List[int]]:
        """
        Recommends videos based on similar authors.

        Args:
            author_id (int): The author ID for which to find similar authors and recommend videos.

        Returns:
            Optional[List[int]]: A list of recommended video IDs, or None if author_id not found.
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained.")

        if author_id not in self.author_to_idx:
            return None

        author_index = self.author_to_idx[author_id]

        # Get similar authors
        similar_author_indices, _ = self.model.similar_items(
            author_index, N=self.num_recommendations
        )

        similar_author_ids = [self.idx_to_author[idx] for idx in similar_author_indices]

        # Filter videos by similar authors
        similar_author_videos = self.video_stats.filter(
            pl.col("author_id").is_in(similar_author_ids)
        )

        # Get top videos per author
        recommended_video_ids = self._get_top_videos(similar_author_videos, n_top=2)

        return recommended_video_ids

    def _get_top_videos(self, data: pl.DataFrame, n_top: int) -> List[int]:
        """Retrieves top n videos for each author based on likes."""
        top_videos = (
            data.sort("v_likes", reverse=True)
            .groupby("author_id")
            .head(n_top)
        )
        return top_videos["video_id"].to_list()

    def evaluate_model(self) -> dict:
        """
        Evaluates the model using precision and MAP metrics.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained.")

        precisions = []
        map_scores = []

        num_authors = self.user_author_matrix.shape[1]

        for author_idx in tqdm(range(num_authors), desc="Evaluating model"):
            # Get the actual users who interacted with this author
            true_user_indices = self.user_author_matrix[:, author_idx].nonzero()[0]

            if len(true_user_indices) == 0:
                continue

            # Get similar authors
            similar_author_indices, _ = self.model.similar_items(
                author_idx, N=self.num_recommendations
            )

            # For evaluation, compare the similar authors with true authors
            hits = len(set(similar_author_indices) & set(true_user_indices))

            precision_at_n = hits / len(similar_author_indices) if len(similar_author_indices) > 0 else 0
            precisions.append(precision_at_n)

            # Compute MAP
            relevant_flags = [1 if idx in true_user_indices else 0 for idx in similar_author_indices]
            if np.sum(relevant_flags) > 0:
                map_score = average_precision_score(relevant_flags, relevant_flags)
                map_scores.append(map_score)
            else:
                map_scores.append(0)

        # Compute average metrics
        precision = np.mean(precisions)
        map_score = np.mean(map_scores)

        return {
            'precision': precision,
            'map_score': map_score
        }


def main():
    parser = argparse.ArgumentParser(description='Train an ALS model and recommend videos based on similar authors.')
    parser.add_argument('--video_stats_path', type=str, required=True, help='Path to the video statistics Parquet file.')
    parser.add_argument('--logs_path_pattern', type=str, required=True, help='Glob pattern to the logs Parquet files.')
    parser.add_argument('--min_video_watch_fraction', type=float, default=0.00008, help='Minimum watch fraction to filter videos.')
    parser.add_argument('--min_videos_per_author', type=int, default=10, help='Minimum number of videos per author.')
    parser.add_argument('--min_user_interactions', type=int, default=100, help='Minimum number of interactions per user.')
    parser.add_argument('--factors', type=int, default=128, help='Number of latent factors in the ALS model.')
    parser.add_argument('--regularization', type=float, default=0.1, help='Regularization parameter for ALS.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations to train the ALS model.')
    parser.add_argument('--num_recommendations', type=int, default=10, help='Number of videos to recommend.')
    parser.add_argument('--model_path', type=str, default='ALS_model.npz', help='Path to save or load the ALS model.')
    parser.add_argument('--author_id', type=int, help='Author ID for which to recommend videos.')
    parser.add_argument('--train', action='store_true', help='Flag to train the model.')
    parser.add_argument('--evaluate', action='store_true', help='Flag to evaluate the model.')
    args = parser.parse_args()

    recommender = AuthorVideoALSRecommender(
        video_stats_path=args.video_stats_path,
        logs_path_pattern=args.logs_path_pattern,
        min_video_watch_fraction=args.min_video_watch_fraction,
        min_videos_per_author=args.min_videos_per_author,
        min_user_interactions=args.min_user_interactions,
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        num_recommendations=args.num_recommendations,
        model_path=args.model_path
    )

    recommender.load_video_stats()
    recommender.prepare_user_author_matrix()

    if args.train:
        recommender.train_model()
    else:
        recommender.load_model()

    if args.evaluate:
        metrics = recommender.evaluate_model()
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Mean Average Precision (MAP): {metrics['map_score']:.4f}")

    if args.author_id is not None:
        recommended_videos = recommender.recommend_videos(author_id=args.author_id)
        if recommended_videos is not None:
            print(f"Recommended videos for author {args.author_id}: {recommended_videos}")
        else:
            print(f"Author ID {args.author_id} not found in the model.")


if __name__ == '__main__':
    main()
