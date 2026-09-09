from pprint import pprint

from src.DataProcessor import load_data_from_csv, get_df_by_platform_product, get_top_and_low_views_from_df



# main entry point.
def main():
    path = "/home/simongalaxy2011/sentimental_analysis/data/amazon_google_reviews.csv"
    df = load_data_from_csv(path=path)

    dataprofiles = get_df_by_platform_product(df=df)

    for dataprofile in dataprofiles:
        dataprofile.filtered_views = get_top_and_low_views_from_df(df=dataprofile.filtered_df, items=20)
        pprint(dataprofile.model_dump())



    return


if __name__ == "__main__":
    main()