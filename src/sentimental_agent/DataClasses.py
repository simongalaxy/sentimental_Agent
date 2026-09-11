from datetime import date
from typing import List, Literal, Optional
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, AfterValidator
from typing_extensions import Annotated

class Category(BaseModel):
    category: List[str]
    
class DataProfile(BaseModel):
    # This configuration tells Pydantic it's okay to accept complex types like DataFrames
    model_config = ConfigDict(arbitrary_types_allowed=True)

    platform: str
    product: str
    filtered_df: Optional[pd.DataFrame] = None
    filtered_views: Optional[List[str]] = None
    category: Optional[List[str]] = None
    processed_df: Optional[pd.DataFrame] = None


### ClassifiedView
def validate_category(values: List[str], info: ValidationInfo) -> List[str]:
        # Extract the allowed categories from the validation context
        allowed_category = info.context.get("allowed_category", [])

        # Check if any item in the input list is not in the allowed categories
        invalid_items = [item for item in values if item not in allowed_category]
        if invalid_items:
            raise ValueError(
                f"Invalid category items: {invalid_items}. Allowed: {allowed_category}"
            )

        return values

class ClassifiedView(BaseModel):
    sentiment: Literal["Positive", "Neutral", "Negative"] = Field(description="Sentiment of the views. It can only be either Positive, Neutral, or Negative.")
    # Annotate the field with our dynamic validator
    category: Annotated[List[str], AfterValidator(validate_category)]
