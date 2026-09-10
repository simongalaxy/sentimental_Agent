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
    categories: Optional[List[str]] = None
    processed_df: Optional[pd.DataFrame] = None


### ClassifiedView
def validate_categories(values: List[str], info: ValidationInfo) -> List[str]:
    # Extract the allowed categories from the validation context
    allowed_categories = info.context.get("allowed_categories", [])

    # Check if any item in the input list is not in the allowed categories
    invalid_items = [item for item in values if item not in allowed_categories]
    if invalid_items:
        raise ValueError(
            f"Invalid categories: {invalid_items}. Allowed: {allowed_categories}"
        )

    return values

class ClassifiedView(BaseModel):
    sentimental: Literal["Positive", "Neutral", "Negative"] = Field("Sentimental of the views. It can only either be Positive, Neutral or Negative.")
     # Annotate the field with our dynamic validator
    categories: Annotated[List[str], AfterValidator(validate_categories)]


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