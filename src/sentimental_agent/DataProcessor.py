import pandas as pd
import os
from typing import List

from src.sentimental_agent.DataClasses import DataProfile

### customs function.
def load_data_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_df_by_platform_product(df: pd.DataFrame) -> List[DataProfile]:
    df_unique_platform_products = df[['Platform', 'Product/Location_Name']].drop_duplicates()
    
    dataprofiles = []
    for platform in df['Platform'].unique():
        for product in df[df['Platform'] == platform]['Product/Location_Name'].unique():
            profile = DataProfile(platform=platform, product=product)
            profile.filtered_df = df[(df['Platform'] == platform) & (df['Product/Location_Name'] == product)]
            dataprofiles.append(profile)
    
    return dataprofiles


def get_top_and_low_views_from_df(df: pd.DataFrame, items: int) -> List[str]:
    # filtered_df = df[df["Rating"].isin([1, 5])].nlargest(20, "Helpful_Votes")
    top_rated_df = df[df["Rating"] == 5].nlargest(items, "Helpful_Votes")
    low_rated_df = df[df["Rating"] == 1].nlargest(items, "Helpful_Votes")
    mid_rated_df = df[~df["Rating"].isin([1,5])].sample(n=items, random_state=42)
    
    filtered_df = pd.concat([top_rated_df, low_rated_df, mid_rated_df], ignore_index=True)
    filtered_df.to_excel("filtered_views.xlsx")
    filtered_views = filtered_df['Review_Text'].tolist()

    return filtered_views
