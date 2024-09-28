import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import json

df = pd.read_parquet('/kaggle/input/video-stat/video_stat.parquet')

df['sentence'] = df['title'] + " " + df['description']

model = SentenceTransformer('intfloat/multilingual-e5-base')
model.to('cuda')
model.max_seq_length = 256

embeddings = model.encode(df['sentence'].values, show_progress_bar=True, device='cuda', batch_size=512)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, 'sbert_video_index.faiss')

k = 300
distances, indices = index.search(embeddings, k)

results = {
    "video_id": [],
    "neighbor_video_id": [],
    "distance": []
}

for i, video_id in enumerate(df['video_id'].values):
    for j in range(k):
        results["video_id"].append(video_id)
        results["neighbor_video_id"].append(df['video_id'].values[indices[i, j]])
        results["distance"].append(distances[i, j])

neighbors_df = pd.DataFrame(results)
neighbors_df.to_csv("sbert_neighbors.csv")

video_neighbors_dict = {}
for video_id, group in neighbors_df.groupby('video_id'):
    sorted_neighbors = group.sort_values(by='distance')['neighbor_video_id'].tolist()
    video_neighbors_dict[video_id] = sorted_neighbors

with open('sbert_neighbors.json', 'w') as json_file:
    json.dump(video_neighbors_dict, json_file, indent=4)