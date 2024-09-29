import argparse
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import json

class VideoTextNeighborFinder:
    """Class to find and save video neighbors based on sentence embeddings."""

    def __init__(self, parquet_path, model_name, max_seq_len, device):
        """Initializes the VideoNeighborFinder with the specified parameters.

        Args:
            parquet_path (str): Path to the parquet file containing video data.
            model_name (str): Name of the SentenceTransformer model to use.
            max_seq_len (int): Maximum sequence length for the model.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.parquet_path = parquet_path
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.device = device
        self.df = None
        self.embeddings = None
        self.index = None
        self.results = None
        self.video_neighbors_dict = None

    def load_data(self):
        """Loads data from the parquet file and prepares sentences."""
        self.df = pd.read_parquet(self.parquet_path)
        self.df['sentence'] = self.df['title'] + " " + self.df['description']

    def compute_embeddings(self):
        """Computes embeddings for the sentences using the specified model."""
        model = SentenceTransformer(self.model_name)
        model.to(self.device)
        model.max_seq_length = self.max_seq_len
        self.embeddings = model.encode(
            self.df['sentence'].values,
            show_progress_bar=True,
            device=self.device,
            batch_size=512
        )

    def build_index(self):
        """Builds a Faiss index from the embeddings and saves it."""
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        faiss.write_index(self.index, 'sbert_video_index.faiss')

    def search_neighbors(self, k):
        """Searches for k nearest neighbors in the index.

        Args:
            k (int): Number of nearest neighbors to find.
        """
        distances, indices = self.index.search(self.embeddings, k)
        self.results = {
            "video_id": [],
            "neighbor_video_id": [],
            "distance": []
        }
        for i, video_id in enumerate(self.df['video_id'].values):
            for j in range(k):
                self.results["video_id"].append(video_id)
                self.results["neighbor_video_id"].append(
                    self.df['video_id'].values[indices[i, j]]
                )
                self.results["distance"].append(distances[i, j])

    def save_results(self):
        """Saves the results to CSV and JSON files."""
        neighbors_df = pd.DataFrame(self.results)
        neighbors_df.to_csv("sbert_neighbors.csv", index=False)
        self.video_neighbors_dict = {}
        for video_id, group in neighbors_df.groupby('video_id'):
            sorted_neighbors = group.sort_values(by='distance')['neighbor_video_id'].tolist()
            self.video_neighbors_dict[video_id] = sorted_neighbors
        with open('sbert_neighbors.json', 'w') as json_file:
            json.dump(self.video_neighbors_dict, json_file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Find video neighbors using embeddings.')
    parser.add_argument('--parquet_path', type=str, required=True, help='Path to the parquet file.')
    parser.add_argument('--model_name', type=str, default='intfloat/multilingual-e5-base', help='Name of the SentenceTransformer model.')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Maximum sequence length for the model.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on.')
    parser.add_argument('--k', type=int, default=300, help='Number of nearest neighbors to search for.')
    args = parser.parse_args()

    finder = VideoTextNeighborFinder(
        parquet_path=args.parquet_path,
        model_name=args.model_name,
        max_seq_len=args.max_seq_len,
        device=args.device
    )
    finder.load_data()
    finder.compute_embeddings()
    finder.build_index()
    finder.search_neighbors(k=args.k)
    finder.save_results()

if __name__ == "__main__":
    main()
