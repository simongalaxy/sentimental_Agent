import pandas as pd
import os
from typing import List
from pprint import pformat

from src.sentimental_agent.DataClasses import DataProfile
from src.sentimental_agent.logger import Logger


### customs function.
def load_data_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in df.columns:
        if df[column].dtype == "str":
            df[column] = df[column].astype(str).str.strip()
        else:
            df[column] = df[column].astype(int)
    
    return df


def get_df_unique_platform_products(df: pd.DataFrame) -> pd.DataFrame:
    return df[['Platform', 'Product/Location_Name']].drop_duplicates()


def get_df_unique_platform(df: pd.DataFrame) -> List[str]:
    return df['Platform'].unique()


def get_df_unique_products_by_platform(df: pd.DataFrame, platform: str) -> List[str]:
    return df[df["Platform"] == platform]['Product/Location_Name'].unique()


def filter_df_by_platform_product(df: pd.DataFrame, platform: str, product: str) -> pd.DataFrame:
    return df[(df["Platform"] == platform) & (df["Product/Location_Name"] == product)].sort_values(by="Rating", ascending=False)

# unique review text from df.
def get_unique_review_texts_from_df(df: pd.DataFrame) -> pd.DataFrame:
    # return df[['Product/Location_Name', 'Review_Text']].drop_duplicates().sort_values(by='Review_Text', ascending=True, ignore_index=True)
    return df.groupby(['Platform', 'Product/Location_Name', 'Review_Text']).agg(Total_Reviews=('Rating', 'count'), Avg_Rating=('Rating', 'mean'), Total_Helpful_Votes=('Helpful_Votes', 'sum'), ).reset_index()

def get_sample_views_from_df(df: pd.DataFrame, items: int) -> List[str]:
    # filtered_df = df[df["Rating"].isin([1, 5])].nlargest(20, "Helpful_Votes")
    top_rated_df = df[df["Rating"] == 5].nlargest(items, "Helpful_Votes")
    low_rated_df = df[df["Rating"] == 1].nlargest(items, "Helpful_Votes")
    mid_rated_df = df[~df["Rating"].isin([1,5])].sample(n=items, random_state=42)
    
    sample_df = pd.concat([top_rated_df, low_rated_df, mid_rated_df], ignore_index=True)
    sample_df.to_excel("filtered_views.xlsx")
    sample_views = sample_df['Review_Text'].tolist()

    return sample_views


def get_dataprofiles_by_platforms_products(df: pd.DataFrame, logger: Logger) -> List[DataProfile]:
    dataprofiles = []
    for platform in get_df_unique_platform(df=df):
        for product in get_df_unique_products_by_platform(df=df, platform=platform):
            logger.info(f"Start generating dataprofile - Platform: {platform}, Product/Location: {product}")
            filtered_df = filter_df_by_platform_product(df=df, platform=platform, product=product)
            unique_views = get_unique_review_texts_from_df(df=filtered_df)
            sample_views = get_sample_views_from_df(df=filtered_df, items=20)
            
            # generate a dataprofile
            profile = DataProfile(
                platform=platform, 
                product=product,
                filtered_df=filtered_df,
                unique_views=unique_views,
                sample_views=sample_views
            )
            dataprofiles.append(profile)
        
            logger.info(f"End generating dataprofile - Platform: {platform}, Product/Location: {product}")
            
    logger.info(f"Total {len(dataprofiles)} dataprofiles generated.\n")
    logger.info("*"*50)
    logger.info(f"Sample dataprofile: {pformat(dataprofiles[1], indent=4)}\n")

    return dataprofiles


def save_processed_df(dataprofile: DataProfile) -> None:
    filename = f"{dataprofile.platform}-{dataprofile.product}.xlsx"
    path = os.path.join(f"{os.getcwd()}/processed_data", filename)
    dataprofile.processed_df.to_excel(path)
    
    return


def transform_str_to_list(input_str: str) -> List[int]:
    return [int(item.strip()) -1 for item in input_str.split(",")]