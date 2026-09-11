import os
import asyncio
from pprint import pprint
import pandas as pd

from src.sentimental_agent.DataProcessor import load_data_from_csv, get_df_by_platform_product, get_top_and_low_views_from_df
from src.sentimental_agent.logger import Logger
from src.sentimental_agent.Settings import settings
from src.sentimental_agent.LLMAgent import LLMAgent



# main entry point.
async def main():
    # initiate settings.
    logger = Logger(__name__).get_logger()
    agent = LLMAgent(logger=logger)

    df = load_data_from_csv(path=os.path.join(f"{os.getcwd()}/data", settings.filename))
    logger.info(f"Data loaded from {settings.filename} with shape: {df.shape}")

    dataprofiles = get_df_by_platform_product(df=df)
    logger.info(f"Number of unique platform-product combinations found: {len(dataprofiles)}")

    for dataprofile in dataprofiles:
        # generate Catagories for each product/service.
        dataprofile.filtered_views = get_top_and_low_views_from_df(df=dataprofile.filtered_df, items=20)
        logger.info(f"Processed platform-product: {dataprofile.platform}-{dataprofile.product} with {len(dataprofile.filtered_views)} filtered views.")
        dataprofile.category = await agent.generate_category(dataprofile=dataprofile)

        # categorize and assign sentimental for each views.
        dataprofile.processed_df = await agent.categorize_all_views(dataprofile=dataprofile)



    return


if __name__ == "__main__":
    asyncio.run(main())