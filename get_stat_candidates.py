import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import timedelta
import faiss
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
import json


df = pd.read_parquet("video_stat.parquet")


# Обработка категориальных фичей
categorical_features = ['category_id']
numerical_features = [
    'v_total_comments', 'v_year_views', 'v_month_views', 'v_week_views', 'v_day_views',
    'v_likes', 'v_dislikes', 'v_duration', 'v_cr_click_like_7_days',
    'v_cr_click_dislike_7_days', 'v_cr_click_vtop_7_days', 'v_cr_click_long_view_7_days',
    'v_cr_click_comment_7_days', 'v_cr_click_like_30_days', 'v_cr_click_dislike_30_days',
    'v_cr_click_vtop_30_days', 'v_cr_click_long_view_30_days', 'v_cr_click_comment_30_days',
    'v_cr_click_like_1_days', 'v_cr_click_dislike_1_days', 'v_cr_click_vtop_1_days',
    'v_cr_click_long_view_1_days', 'v_cr_click_comment_1_days', 'v_avg_watchtime_1_day',
    'v_avg_watchtime_7_day', 'v_avg_watchtime_30_day', 'v_frac_avg_watchtime_1_day_duration',
    'v_frac_avg_watchtime_7_day_duration', 'v_frac_avg_watchtime_30_day_duration',
    'v_category_popularity_percent_7_days', 'v_category_popularity_percent_30_days',
    'v_long_views_1_days', 'v_long_views_7_days', 'v_long_views_30_days'
]


log_features = ['v_year_views', 'v_month_views', 'v_week_views', 'v_day_views']
df[log_features] = df[log_features].apply(lambda x: np.log1p(x))


# Все числовые данные нормализуем
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Обработка категориальных фичей с помощью OneHotEncoding
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
categorical_data = encoder.fit_transform(df[categorical_features])


encoded_df = np.hstack((df[numerical_features], df[log_features], categorical_data))

encoded_df = encoded_df.astype('float32')

index = faiss.IndexFlatL2(encoded_df.shape[1])
index.add(encoded_df)

faiss.write_index(index, 'stat_video_index.faiss')



k = 300
distances, indices = index.search(encoded_df, k)


# Подготовка результатов для нового DataFrame
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

neighbors_df.to_csv("stat_neighbors.csv")

video_neighbors_dict = {}

for video_id, group in neighbors_df.groupby('video_id'):
    sorted_neighbors = group.sort_values(by='distance')['neighbor_video_id'].tolist()
    video_neighbors_dict[video_id] = sorted_neighbors

# Сохранение в JSON файл
with open('stat_neighbors.json', 'w') as json_file:
    json.dump(video_neighbors_dict, json_file, indent=4)

