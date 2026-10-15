import os
import pandas as pd
from typing import List
from pprint import pformat

from src.sentimental_agent.Utilities.Logger import Logger


def get_filePath(filename: str, folderpath: str, logger: Logger) -> str:
    """
    Get the file path of the current script.

    Returns:
        str: The file path of the current script.
    """
    filepath = os.path.join(folderpath, filename)
    logger.info(f"File path: {filepath}")
    
    return filepath


def load_csv(filepath: str, logger: Logger) -> pd.DataFrame | None:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    try:
        df = pd.read_csv(filepath)
        logger.info("Data loaded successfully.")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame preview:\n{df.head(5)}")
        logger.info("#" * 50)
        return df
    except Exception as e:
        logger.error(f"An error occurred while loading the CSV file: {e}")
        raise RuntimeError(f"An error occurred while loading the CSV file: {e}")
        return


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # data cleansing and set datatype.
    
    # 1. trim string columns
    str_columns = ["Platform", "Product/Location_Name", "Review_Text"]
    df[str_columns] = df[str_columns].apply(lambda x: x.str.strip())

    # 2. set up date datatype.
    df["Date"] = df["Date"].apply(pd.to_datetime, dayfirst=False)

    return df


def get_unique_from_column(df: pd.DataFrame, col_name: str) -> List[str]:

    return df[col_name].unique().tolist()


def get_df_by_platform_product(df: pd.DataFrame, temp_folder_path: str, logger: Logger) -> List[pd.DataFrame]:

    # get unique platforms.
    platforms = get_unique_from_column(df=df, col_name="Platform")

    # prepare splitted csv for each products and consolidated duplicated review_texts.
    dfs = []
    for platform in platforms:
        for product in df[df["Platform"] == platform]["Product/Location_Name"].unique().tolist():
            df_by_product = df[(df["Platform"] == platform) & (df["Product/Location_Name"] == product)]\
                            .groupby(by=["Platform", "Product/Location_Name", "Review_Text"])\
                            .agg(Total_Reviews=("Review_ID", "count"), Average_Rating=("Rating", "mean"), Total_Helpful_Votes=("Helpful_Votes", "sum"))\
                            .reset_index()
            filename = f"{platform.replace(" ", "_")}_{product.replace(" ", "_")}.csv"
            dfs.append(df_by_product)
            df_by_product.to_csv(os.path.join(temp_folder_path, filename))
            logger.info(f"{filename} created.")
    
    logger.info(f"total csv created: {len(dfs)}")
    

    return dfs

