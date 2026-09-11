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

    

# # --- How to use it at runtime ---

# # 1. Define your dynamic list of allowed categories
# current_categories = ["electronics", "clothing", "home"]

# # 2. Pass the list inside the `context` dictionary during validation
# try:
#     # This will PASS
#     valid_item = ItemClassifier.model_validate(
#         {"name": "Laptop", "categories": ["electronics", "home"]},
#         context={"allowed_categories": current_categories},
#     )
#     print("Success:", valid_item)

#     # This will FAIL because "food" is not in our dynamic list
#     invalid_item = ItemClassifier.model_validate(
#         {"name": "Apple", "categories": ["food", "electronics"]},
#         context={"allowed_categories": current_categories},
#     )
# except ValidationError as e:
#     print("\nValidation Error caught successfully:")
#     print(e)