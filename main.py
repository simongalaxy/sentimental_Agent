import os
import asyncio
from pprint import pprint
import pandas as pd

from src.sentimental_agent.logger import Logger
from src.sentimental_agent.Settings import settings
from src.sentimental_agent.LLMAgent import LLMAgent
from src.sentimental_agent.DataProcessor import (
    load_data_from_csv, 
    get_dataprofiles_by_platforms_products, 
    save_processed_df,
)

# main entry point.
async def main():
# def main():
    # initiate settings.
    logger = Logger(__name__).get_logger()
    agent = LLMAgent(logger=logger)

    # load datasource.
    df = load_data_from_csv(path=os.path.join(f"{os.getcwd()}/data", settings.filename))
    logger.info(f"Data info: \n%s", df.info())
    logger.info(f"Data loaded from {settings.filename} with shape: {df.shape}")

    # generate dataprofiles by platforms by products.
    dataprofiles = get_dataprofiles_by_platforms_products(df=df, logger=logger)
    
    # generate categories and categorize unique views.abs
    for i, dataprofile in enumerate(dataprofiles[2:3], start=1):
        logger.info(f"No.{i} of {len(dataprofiles)}:")
        logger.info(f"Platform: {dataprofile.platform}")
        logger.info(f"Product: {dataprofile.product}")
        logger.info(f"Dimension of filtered_df: {dataprofile.filtered_df.shape}")
        
        # generate Catagories for each product/service.
        dataprofile.category = await agent.generate_category(dataprofile=dataprofile)

        # categorize and assign sentimental for each views.
        # dataprofile.processed_df = await agent.categorize_all_views(dataprofile=dataprofile)
        # save_processed_df(dataprofile=dataprofile)
        # logger.info(f"Processed_df saved: {dataprofile.platform}-{dataprofile.product}")
        async for views_batch in agent.categorize_all_views(dataprofile=dataprofile):
            for view in views_batch:
                logger.info(f"processed view: {pformat(view, indent=4)}")



    return


if __name__ == "__main__":
    asyncio.run(main())
    # main()