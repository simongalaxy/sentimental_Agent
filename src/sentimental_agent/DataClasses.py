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
    sample_views: Optional[List[str]] = None
    unique_views: Optional[pd.DataFrame] = None
    filtered_views: Optional[List[str]] = None
    category: Optional[List[str]] = None
    processed_df: Optional[pd.DataFrame] = None


### ClassifiedView
def validate_category(value: str, info: ValidationInfo) -> str:
    # Extract the allowed categories from the validation context
    allowed_category = info.context.get("allowed_category", [])
    
    # Check if the input string is in the allowed categories
    if value not in allowed_category:
        raise ValueError(
            f"Invalid category item: '{value}'. Allowed: {allowed_category}"
        )
    return value

class ClassifiedView(BaseModel):
    sentiment: Literal["Positive", "Neutral", "Negative"] = Field(description="Sentiment of the views. It can only be either Positive, Neutral, or Negative.")
    # Annotate the field with our dynamic validator
    category: Annotated[str, AfterValidator(validate_category)]

