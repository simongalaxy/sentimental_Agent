import os
from pprint import pprint

from src.sentimental_agent.DataProcessor import load_data_from_csv, get_df_by_platform_product, get_top_and_low_views_from_df
from src.sentimental_agent.logger import Logger
from src.sentimental_agent.Settings import settings



# main entry point.
def main():
    # initiate settings.
    logger = Logger(__name__).get_logger()

    df = load_data_from_csv(path=os.path.join(f"{os.getcwd()}/data", settings.filename))
    logger.info(f"Data loaded from {settings.filename} with shape: {df.shape}")

    dataprofiles = get_df_by_platform_product(df=df)
    logger.info(f"Number of unique platform-product combinations found: {len(dataprofiles)}")

    for dataprofile in dataprofiles:
        dataprofile.filtered_views = get_top_and_low_views_from_df(df=dataprofile.filtered_df, items=20)
        logger.info(f"Processed platform-product: {dataprofile.platform}-{dataprofile.product} with {len(dataprofile.filtered_views)} filtered views.")
    



    return


if __name__ == "__main__":
    main()