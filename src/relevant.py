import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
from pydantic import BaseModel
from typing import List

class Item(BaseModel):
    name: str
    description: str | None
    category:str | None
    video_id: str
    interaction: int = 0


class CandidatesSelector:
    
    """A class to select candidates based on video relevance scores.

    This class computes relevance scores for videos based on daily, weekly, 
    and monthly views, and allows for selection of candidates from different categories.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing video statistics.
    """
    
    def __init__(self):
        """
            Initializes CandidatesSelector with video data.
        """
        
        self.category_weights = {}
        self.hidden_ids = set()


    def get_relevance_score(self, data):
        """Calculates the relevance scores for videos.

        The relevance score is computed as a weighted sum of the scaled daily, 
        weekly, and monthly views.

        Returns:
            pd.DataFrame: DataFrame with an additional 'relevance_score' column,
            sorted by relevance score in descending order.
        """
        
        relevant_features = [
            'v_day_views',
            'v_week_views',
            'v_month_views',
        ]
        scaled_relevant_features = [feature + '_scaled' for feature in relevant_features]

        scaler = MinMaxScaler()
        for feature in relevant_features:
            data[feature + '_scaled'] = scaler.fit_transform(data[[feature]])

        # Calculate relevance score as a weighted sum of features
        data['relevance_score'] = (
            data['v_day_views_scaled'] * 0.55 + 
            data['v_week_views_scaled'] * 0.35 + 
            data['v_month_views_scaled'] * 0.1  
        )

        data = data.sort_values(by='relevance_score', ascending=False)
        data.drop(scaled_relevant_features, axis=1, inplace=True)

        return data


    def get_top_n_per_category_candidates(self, data, n=100):
        """Retrieves the top N candidates per category based on relevance score.

        Args:
            n (int): The number of top candidates to retrieve per category.

        Returns:
            pd.DataFrame: DataFrame containing the top N candidates for each category.
        """
        
        data = data.sort_values(by=['category_id', 'relevance_score'], ascending=[True, False])
        top_n_per_category = data.groupby('category_id').head(n)
        
        return top_n_per_category


    def choose_candidates_from_categories(self, top_category_candidates, weighted_chosen_categories):
        """Chooses candidates from specified categories.

        Args:
            top_category_candidates (pd.DataFrame): DataFrame containing candidates 
                sorted by relevance score.
            weighted_chosen_categories (list): List of categories to choose candidates from.

        Returns:
            pd.DataFrame: DataFrame containing one candidate from each chosen category.
        """
        
        candidates = pd.concat(
            [top_category_candidates[top_category_candidates['category_id'] == i].sample(n=1) for i in weighted_chosen_categories]
        )

        return candidates

    def convert_candidates_to_Item_class(self, final_candidates):
        """Convert rows of a DataFrame to a list of Item class instances.
    
        Args:
            final_candidates (pd.DataFrame): DataFrame containing video information with
                columns 'title', 'v_pub_datetime', 'description', and 'video_id'.
    
        Returns:
            List[Item]: A list of Item instances created from the DataFrame rows.
        """
        items: List[Item] = []
        for _, row in final_candidates.iterrows():
            item = Item(
                name=row['title'],
                description= row['description'] if pd.notna(row['description']) else None,
                video_id=row['video_id'],
                category=row['category_id'],
                interaction=0
            )
            items.append(item)

        return items


    def update_category_weights(self, items):
        """Update weights for categories

        This method ensures that the selected candidates are unique in terms of 
        authors and titles, and uses weighted random selection for category choices.

        Args:
            List[item]: A list of Item instances.

        Returns:
            None
        """
        for item in items:
            if item.interaction == 0:
                selector.category_weights[item.category] *= 0.8
            elif item.interaction == -1:
                selector.category_weights[item.category] *= 0.2
            elif item.interaction == 1:
                selector.category_weights[item.category] *= 1.5


    def get_n_candidates(self, top_category_candidates, n=10):
        """Selects a specified number of candidates based on their categories.

        This method ensures that the selected candidates are unique in terms of 
        authors and titles, and uses weighted random selection for category choices.

        Args:
            top_category_candidates (pd.DataFrame): DataFrame containing the top candidates 
                per category with their relevance scores.
            n (int): The number of unique candidates to select.

        Returns:
            List[item]: A list of Item instances created from the DataFrame rows.
        """
        
        # Calculate mean relevance score per category
        top_category_candidates['mean_relevance_score_per_category'] = top_category_candidates.groupby('category_id')['relevance_score'].transform('mean')
        top_category_candidates = top_category_candidates.sort_values(by=['mean_relevance_score_per_category', 'category_id'], ascending=[False, False])

        k = 0

        # Get category weights 
        if not self.category_weights:
            category_weights = dict(zip(top_category_candidates['category_id'].unique(), top_category_candidates['mean_relevance_score_per_category'].unique()))
        else:
            category_weights = self.category_weights
        
        self.category_weights = category_weights
        # Randomly-weighted selection of categories
        weighted_chosen_categories = random.choices(list(category_weights.keys()), weights=list(category_weights.values()), k=n)

        # Choose from the most popular videos without duplicates
        top_category_candidates = top_category_candidates.sample(frac=1).reset_index(drop=True)
        top_category_candidates = top_category_candidates[~top_category_candidates['video_id'].isin(self.hidden_ids)]
        top_category_candidates.drop_duplicates(subset='author_id', inplace=True)
        top_category_candidates.drop_duplicates(subset='title', inplace=True)

        candidates = pd.DataFrame()
        
        while (
            (len(candidates) < n 
            or candidates['author_id'].nunique() < n 
            or any(item in self.hidden_ids for item in candidates['video_id'].values)
            )
            and k <= 30
        ):
            candidates = self.choose_candidates_from_categories(
                top_category_candidates, 
                weighted_chosen_categories
            )
            k += 1


        self.hidden_ids.update(list(candidates['video_id'].values))

        # convert items-candidates to BaseClass
        candidates = self.convert_candidates_to_Item_class(candidates)
        
        return candidates


