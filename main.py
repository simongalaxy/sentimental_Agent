import os
import asyncio
from pprint import pprint
import pandas as pd

from src.sentimental_agent.DataProcessor import load_data_from_csv, get_df_by_platform_product, get_top_and_low_views_from_df, save_processed_df, get_df_unique_platform, get_df_unique_products_by_platform, get_unique_review_texts_from_df, filter_df_by_platform_product, get_unique_review_texts_from_df
from src.sentimental_agent.logger import Logger
from src.sentimental_agent.Settings import settings
from src.sentimental_agent.LLMAgent import LLMAgent



# main entry point.
# async def main():
def main():
    # initiate settings.
    logger = Logger(__name__).get_logger()
    agent = LLMAgent(logger=logger)

    # load datasource.
    df = load_data_from_csv(path=os.path.join(f"{os.getcwd()}/data", settings.filename))
    logger.info(f"Data info: \n%s", df.info())
    logger.info(f"Data loaded from {settings.filename} with shape: {df.shape}")

    # get the unique platform and its associated products/locations for filtering the dataframe.
    unique_platform = get_df_unique_platform(df=df)
    for i, item in enumerate(unique_platform, start=1):
        logger.info(f"No.{i} - Platform: {item}")
        
        unique_products_by_platform = get_df_unique_products_by_platform(df=df, platform=item)
        logger.info(f"unique products: {unique_products_by_platform}")


    filtered_df = filter_df_by_platform_product(df=df, platform="Amazon", product="Atomic Habits by James Clear")
    unique_review_texts = get_unique_review_texts_from_df(df=filtered_df)

    logger.info(f"unique_review_texts: {unique_review_texts}")


    
    

    # dataprofiles = get_df_by_platform_product(df=df)
    # logger.info(f"Number of unique platform-product combinations found: {len(dataprofiles)}")

    # for i, dataprofile in enumerate(dataprofiles, start=1):
    #     logger.info(f"No.{i} of {len(dataprofiles)}:")
    #     logger.info(f"Platform: {dataprofile.platform}")
    #     logger.info(f"Product: {dataprofile.product}")
    #     logger.info(f"Dimension of filtered_df: {dataprofile.filtered_df.shape}")
        
        # # generate Catagories for each product/service.
        # dataprofile.filtered_views = get_top_and_low_views_from_df(df=dataprofile.filtered_df, items=20)
        # logger.info(f"Processed platform-product: {dataprofile.platform}-{dataprofile.product} with {len(dataprofile.filtered_views)} filtered views.")
        # dataprofile.category = await agent.generate_category(dataprofile=dataprofile)

        # # categorize and assign sentimental for each views.
        # dataprofile.processed_df = await agent.categorize_all_views(dataprofile=dataprofile)
        # save_processed_df(dataprofile=dataprofile)
        # logger.info(f"Processed_df saved: {dataprofile.platform}-{dataprofile.product}")




    return


if __name__ == "__main__":
    # asyncio.run(main())
    main()