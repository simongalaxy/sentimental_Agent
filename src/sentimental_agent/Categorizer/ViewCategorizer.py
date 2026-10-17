import torch
import pandas as pd
from typing import List
from transformers import pipeline

from src.sentimental_agent.Utilities.Logger import Logger
from src.sentimental_agent.Utilities.Settings import settings


class ViewCategorizer:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.categorizer_model = settings.categorizier_model
        self.labeller_model = settings.labeller_model
        

    def _format_reviews(self, review_texts: List[str]) -> str:
        # Convert the list to a single numbered string for the model to read
        formatted_reviews = "\n".join([f"{i+1}. {text}" for i, text in enumerate(review_texts)])

        return formatted_reviews


    def _get_categories_from_reviews(self, review_texts: List[str]) -> List[str]:

        self.logger.info("Loading local Hugging Face Text-generation Model....")

        # Get Formatted reviews.
        formatted_reviews = self._format_reviews(review_texts=review_texts)

        # Format the prompt using standard Chat Templates
        messages = [
            {
                "role": "system",
                "content": "You are a professional product analyst. Your job is to extract clean, high-level category labels from customer feedback."
            },
            {
                "role": "user",
                "content": (
                    f"Analyze this list of unique customer reviews and extract the top 3 to 5 most common subcategories/topics "
                    f"they complain or praise about (e.g., 'Battery Life', 'Customer Service', 'Pricing').\n\n"
                    f"Provide the output strictly as a comma-separated list inside brackets, like this: [Category 1, Category 2]. "
                    f"Do not include any intro, outro, explanations, or markdown code blocks.\n\n"
                    f"Reviews:\n{formatted_reviews}"
                )
            }
        ]

        views_categorizer = pipeline(
                        "text-generation",
                        model=self.categorizer_model,
                        torch_dtype=torch.float32, # Optimal for CPU environments
                        device_map="auto"
                    )

        # 4. Generate the categories
        self.logger.info("Analyzing the list of reviews...")
        outputs = views_categorizer(messages, max_new_tokens=100, return_full_text=False)
        response_text = outputs[0]['generated_text'].strip()

        self.logger.info("\n=== AUTO-EXTRACTED CATEGORIES FROM HUGGING FACE LLM ===")
        categories = [item.strip("[]") for item in response_text.split(",")]
        self.logger.info(f"Categories: {categories}")

        return categories


    def label_views_by_categories(self, df: pd.DataFrame) -> pd.DataFrame:
      
        # 1. Texts to be labeled.
        review_texts = df["Review_Text"].tolist()

        # 2. Get Categories from review texts.
        categories = self._get_categories_from_reviews(review_texts=review_texts)

        # 3. Initialize the zero-shot classification pipeline
        # Uses 'facebook/bart-large-mnli' as a highly capable, default model
        views_labeller = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli"
            )

        # 3. Classify Review texts.
        results = views_labeller(review_texts, candidate_labels=categories, multi_label=False)

        # 4. Extract predictions and add them to the DataFrame
        # We pick the label with the highest confidence score
        top_labels = [res["labels"][0] for res in results]
        # top_scores = [res["scores"][0] for res in results]

        # Create a copy of the dataframe to avoid setting with copy warning
        output_df = df.copy()
        output_df["Category"] = top_labels
        # output_df["confidence_score"] = top_scores
        
        self.logger.info(f"output df: {output_df.head(5)}")
        return output_df