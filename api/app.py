from typing import Dict, List
from fastapi import FastAPI
import numpy as np
from pydantic.class_validators import root_validator
# from src.whatever import make_recommendations
from src.utils import map_article_id_to_article
from pydantic import BaseModel, validator

class UserItemInteraction(BaseModel):
    # type hints
    article_id: List[int]
    click: List[int]
    time_spent: List[int]

    # data validation
    @validator("article_id")
    def article_id_length_must_be_10(cls, value):
        if len(value) != 10 & len(value) != 0:
                raise ValueError(f"Length of `article_id` list must be == 10, you passed a list of length {len(value)}")
        return value

    @validator("click")
    def click_length_must_be_10(cls, value):
        if len(value) != 10 & len(value) != 0:
            raise ValueError(f"Length of `click` list must be == 10, you passed a list of length {len(value)}")
        return value

    @validator("time_spent")
    def time_spent_length_must_be_10(cls, value):
        if len(value) != 10 & len(value) != 0:
            raise ValueError(f"Length of `time_spent` list must be == 10, you passed a list of length {len(value)}")
        return value

    @validator("article_id")
    def article_ids_must_be_positive(cls, value):
        for article_id_ in value:
            if article_id_ <= 0:
                raise ValueError(f"`article_id` must only have positive values, you passed \
{value}, non positive value at index {value.index(article_id_)}")
        return value

    @validator("click")
    def clicks_must_be_positive(cls, value):
        for click_ in value:
            if click_ not in [0, 1]:
                raise ValueError(f"`click` must only have 0 or 1, you passed \
{value}, non [0, 1] value at index {value.index(click_)}")
        return value

    @validator("time_spent")
    def time_spents_must_be_positive(cls, value):
        for time_spent_ in value:
            if time_spent_ < 0:
                raise ValueError(f"`time_spent` must only have positive values, you passed \
{value}, negative value at index {value.index(time_spent_)}")
        return value

    # root-validation occurs after field validation(s)
    @root_validator(pre=False)
    def click_and_time_spent_must_agree(cls, values):
        click = values.get("click")
        time_spent = values.get("time_spent")
        for index_, (click_, time_spent_) in enumerate(zip(click, time_spent)):
            if click_ == 0:
                if time_spent_ > 0:
                    raise ValueError(f"`time_spent` list can't have a non-zero value \
at an index where `click` list has zero, check `click`, and `time_spent` at index {index_}")
        return values

    @validator("article_id")
    def article_id_must_have_unique_values(cls, value):
        if len(value) != len(list(set(value))):
            raise ValueError(f"All the values in `article_id` must be unique, you passed {value}")
        return value

#TODO: remove this once a function to make reocmmendations is made 
def make_recommendations(user_item_interaction):
    # train model with new user_item interactions
    return {i: int(np.random.choice(np.arange(1, 7900))) for i in range(1, 11)}

app = FastAPI()

@app.get("/")
def root():
    return {"message": "📰 we recommend news"}

@app.post("/feed/")
async def get_recommendations(user_item_interaction: UserItemInteraction):
    recommendation_dict = make_recommendations(user_item_interaction)
    recommendation_dict = map_article_id_to_article(
        article_ids=list(recommendation_dict.values()),
        fields=["heading", "article_id"]
    )
    return {
        "recommendations": recommendation_dict
    }