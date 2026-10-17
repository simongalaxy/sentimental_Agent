# sentimental labeling.
import pandas as pd
from transformers import pipeline

from src.sentimental_agent.Utilities.Logger import Logger
from src.sentimental_agent.Utilities.Settings import settings


class SentimentalClassifier:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.model = settings.sentimental_model

        # Load a fast, highly accurate sentiment classifier
        self.classifier = pipeline("sentiment-analysis", model=self.model)

    # Batch process unique reviews
    def get_sentimental_from_review(self, df: pd.DataFrame, review_col: str) -> pd.DataFrame:
        df['Sentiment'] = [res['label'] for res in self.classifier(df[review_col].tolist(), batch_size=32)]
        self.logger.info(f"DataFrame with Sentiment column: \n%s", df.head(5))

        return df
