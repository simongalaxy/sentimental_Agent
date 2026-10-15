import os


from src.sentimental_agent.Utilities.Settings import settings
from src.sentimental_agent.Utilities.Logger import Logger
from src.sentimental_agent.DataProcessor.DataProcessor import (
    get_filePath, 
    load_csv,
    get_df_by_platform_product
)
from src.sentimental_agent.Sentimental import SentimentalClassifer


def main():
    """
    Main function to execute the data processing tasks.
    """
    # initiate instance.
    logger = Logger(__name__).get_logger()
    senti_classifer = SentimentalClassifer(logger=logger)

    # load data source.
    source_filename = settings.source_filename
    source_folderpath = settings.source_folderpath

    # Get the file path of the source CSV file.
    source_filepath = get_filePath(
        filename=source_filename, 
        folderpath=source_folderpath, 
        logger=logger
        )

    # Load the CSV file into a DataFrame.
    df = load_csv(
        filepath=source_filepath, 
        logger=logger
        )

    dfs = get_df_by_platform_product(
        df=df,
        temp_folder_path=settings.tempfiles_folderpath,
        logger=logger
    )

    # get the sentimental from review_text.
    i = 2
    df_test = dfs[i]

    df_sentimental = senti_classifer.get_sentimental_from_review(df=df_test, review_col="Review_Text")



    return


# Entry point of the script.
if __name__ == "__main__":
    main()