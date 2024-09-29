import pandas as pd

from typing import List, Dict
from pydantic import BaseModel
import pandas as pd

class Item(BaseModel):
    """
    Pydantic model representing a video item.
    """
    name: str
    description: str | None
    category: str | None
    video_id: str
    interaction: int = 0


class Reranker:
    def __init__(self, 
                 video_stat: pd.DataFrame,
                 sources: list[dict],
                 sort_col: str = 'v_frac_avg_watchtime_30_day_duration',
                 ascending: bool = False):
        self.video_stat = video_stat
        assert sort_col in video_stat.columns
        self.sources = sources
        self.sort_col = sort_col
        self.ascending = ascending

    def generate(self, items: List[Item], top_k: int = 8) -> List[Item]:
        """
        Generate a list of top K videos based on liked and disliked videos.

        Parameters:
        ----------
        items : List[Item]
            List of items containing liked and disliked video IDs.
        top_k : int, optional
            Number of top videos to return (default is 8).

        Returns:
        ----------
        List[Item]
            List of top K video items sorted by the specified column.
        """
        video_id_liked = [item.video_id for item in items if item.interaction == 1]
        video_id_disliked = [item.video_id for item in items if item.interaction == 0]
        exclude_ids = set(item.video_id for item in items)

        candidate_ids = []
        
        for video_id in video_id_liked:
            for source in self.sources:
                curr_candidates = source.get(video_id, [])
                candidate_ids.extend(curr_candidates)
        
        clean_candidates_ids = set(candidate_ids) - exclude_ids
        df = self.video_stat[self.video_stat['video_id'].isin(clean_candidates_ids)]
        sorted_videos = df.sort_values(by=self.sort_col, ascending=self.ascending).head(top_k)

        result_items = [
            Item(
                name=row['title'],
                description=row['description'],
                category=row["category_id"], 
                video_id=row['video_id'],
                interaction=0, 
            ) for _, row in sorted_videos.iterrows()
        ]
        
        return result_items


# source = json.load(open('video_neighbors.json'))
# top_k = 8
# video_stat = pd.read_parquet('video_stat.parquet')
# sort_col = 'v_frac_avg_watchtime_30_day_duration'
# reranker = Reranker(video_stat, sources=[source])
# items = [
#     Item(name='Video A', description='Desc A', category='Cat A', video_id="31b533da-a57b-4395-8068-1515448b562c", interaction=1),
#     Item(name='Video B', description='Desc B', category='Cat B', video_id="bede0d0a-e203-4fb5-9153-89f214e474f8", interaction=0)
# ]
# result = reranker.generate(items)
# print(result)
