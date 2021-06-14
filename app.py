import streamlit as st
import numpy as np
import pandas as pd

# from src.hybrid_rec_sys import top_10_most_popular
def top_10_most_popular():
    article_ids = np.random.randint(low=1, high=7900, size=10)
    headings = [f"Article #{i+1}" for i in article_ids]
    contents = [f"""this is some sample text for article #{i+1} and
    this is a very long news. with punctuations and stuff.
    you shouldn't be reading this and should follow me on 
    twitter @p69ns instead. satchel out.""" for i in article_ids]
    return pd.DataFrame({
        "article_ids": article_ids,
        "headings": headings,
        "contents": contents,
    })
# ------------------

num_cols = 3
cols = st.beta_columns(num_cols)
cols = cols * (int(10 / num_cols) + 1)
articles = top_10_most_popular()

user_id = 1001 # user_id
# session_id is available
items_time_spent = pd.DataFrame(columns=["session_id", "article_id", "time_spent"])
for id, (article, col) in enumerate(zip(articles.values, cols)):
    col.subheader(f'{article[1]}')
    col.write(article[2])
    time_spent = col.slider(
        label="Time Spent",
        min_value=0,
        max_value=500,
        key=article[0]
    )

next_col = cols[cols.index(col) + 1]
if next_col.button("End Session"):
    # get recommendations for next session
    articles = top_10_most_popular()