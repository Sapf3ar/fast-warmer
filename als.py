import json
import polars as pl
from typing import Optional
from implicit.cpu.als import AlternatingLeastSquares


class ALS:
    def __init__(
        self,
        model_path: str,
        authors_ids_path: str,
        video_stats_path: str,
        num_close_videos: int
    ) -> None:
        """Инициализация модели ALS, загрузка данных по авторам и статистике видео"""
        self.model = AlternatingLeastSquares().load(model_path)

        self.video_stats = pl.read_parquet(video_stats_path)
        self.num_close_videos = num_close_videos
        self.num_close_authors = num_close_videos // 2

        with open(authors_ids_path, "r") as f:
            self.authors_to_idxs = json.load(f)
        ids_to_autors = {}
        for k,v in self.authors_to_idxs.items():
            ids_to_autors[v] = k
        self.ids_to_autors = ids_to_autors

    def __getitem__(self, author_id: int) -> Optional[list[int]]:
        """Возвращает список рекомендованных видео для похожих авторов"""
        if author_id not in self.authors_to_idxs:
            return None

        author_idx = self.authors_to_idxs[author_id]

        # Получаем похожих авторов через ALS модель
        similar_author_idxs = self.model.similar_items(author_idx, N=self.num_close_authors)[0]

        similar_author_ids = []
        for similar_author_idx in similar_author_idxs:
            similar_author_ids.append(self.ids_to_autors[similar_author_idx])
    
        # Фильтруем видео по списку похожих авторов
        close_author_videos = self.video_stats.filter(
            pl.col("author_id").is_in(similar_author_ids)
        )

        # Получаем рекомендованные видео с использованием взвешенного семплера
        recommended_video_ids = get_top_videos(close_author_videos, n_top=2)

        # Возвращаем список video_id
        return recommended_video_ids


def get_top_videos(data, n_top):
    top_videos = (
        data
        .sort("v_likes", descending=True)
        .group_by("author_id")
        .head(n_top)
    )
    return top_videos["video_id"].to_list()
