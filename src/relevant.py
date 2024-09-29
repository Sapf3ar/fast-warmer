import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
from main import Item
from typing import List

class CandidatesSelector:
    
    """A class to select candidates based on video relevance scores.

    This class computes relevance scores for videos based on daily, weekly, 
    and monthly views, and allows for selection of candidates from different categories.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing video statistics.
    """
    
    def __init__(self):
        """Initializes CandidatesSelector with video data.

        Args:
            data (pd.DataFrame): A DataFrame containing the video statistics,
                including view counts and categories.
        """
        
        # self.weighted_chosen_categories


    def get_relevance_score(self):
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
            self.data[feature + '_scaled'] = scaler.fit_transform(self.data[[feature]])

        # Calculate relevance score as a weighted sum of features
        self.data['relevance_score'] = (
            self.data['v_day_views_scaled'] * 0.55 + 
            self.data['v_week_views_scaled'] * 0.35 + 
            self.data['v_month_views_scaled'] * 0.1  
        )

        self.data = self.data.sort_values(by='relevance_score', ascending=False)
        self.data.drop(scaled_relevant_features, axis=1, inplace=True)

        return self.data


    def get_top_n_per_category_candidates(self, n=100):
        """Retrieves the top N candidates per category based on relevance score.

        Args:
            n (int): The number of top candidates to retrieve per category.

        Returns:
            pd.DataFrame: DataFrame containing the top N candidates for each category.
        """
        
        data = self.data.sort_values(by=['category_id', 'relevance_score'], ascending=[True, False])
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
                date=row['v_pub_datetime'],
                description= row['description'] if pd.notna(row['description']) else None,
                video_id=row['video_id']
            )
            items.append(item)

        return items


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

        # Get category weights 
        dict_from_columns = dict(zip(top_category_candidates['category_id'].unique(), top_category_candidates['mean_relevance_score_per_category'].unique()))
        
        # Randomly-weighted selection of categories
        weighted_chosen_categories = random.choices(list(dict_from_columns.keys()), weights=list(dict_from_columns.values()), k=n)

        # Choose from the most popular videos without duplicates
        top_category_candidates = top_category_candidates.sample(frac=1).reset_index(drop=True)
        top_category_candidates.drop_duplicates(subset='author_id', inplace=True)
        top_category_candidates.drop_duplicates(subset='title', inplace=True)

        candidates = pd.DataFrame()
        while len(candidates) < n or candidates['author_id'].nunique() < n:
            candidates = self.choose_candidates_from_categories(top_category_candidates, weighted_chosen_categories)

        print(len(candidates))
        # convert items-candidates to BaseClass
        candidates = self.convert_candidates_to_Item_class(candidates)
        
        return candidates

# 1 - if you want to calculate relevant score and get populer videos for each category
# selector = CandidatesSelector()
# calculate relevance scores
# data_with_scores = selector.get_relevance_score()

# Get top candidates per category
# top_200_per_category = selector.get_top_n_per_category_candidates(n=200)

# Get 10 candidates for cold users
# final_candidates = selector.get_n_candidates(top_100_per_category, n=10)


# 2 - If you already have stats per category for month


