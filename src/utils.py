import logging
import pandas as pd


def create_logger(logger_name):
    logger = logging.getLogger(logger_name)  # should be __name__ when inside a module
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(name)s:%(message)s")
    if not logger.handlers:
        file_handler = logging.FileHandler(f"logs/{logger_name}.log")
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger


def map_article_id_to_article(article_ids, fields):
    article_data = pd.read_csv("data/common/bbc_toi_yahoo_news_clustered_vectored.csv")
    id_to_article = {}
    for article_id in article_ids:
        article = article_data.loc[article_data.loc[:, "article_id"] == article_id, :]
        article_heading_content = {}
        for field in fields:
            field_value = article[field].values.item()
            article_heading_content[field] = field_value
        id_to_article[article_id] = article_heading_content
    return id_to_article

def top_10_recommendations(clickstream_data, article_data):
    """returns a dict with keys 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
       and value = {
           "article_id": int,
           "heading": str,
           "content": str
       }

    Args:
        clickstream_data (pd.DataFrame): df containing all the 
        clickstream encountered so far
        article_data (pd.DataFram): df containing article data
    """
    top_10_most_popular_articles = clickstream_data.loc[:, "article_id"].value_counts()[:10]
    
    top_10_most_popular_article_ids = top_10_most_popular_articles.index
    headings = []
    contents = []
    article_ids = []

    for article_id in top_10_most_popular_article_ids:
        article_ids.append(article_id)
        headings.append(article_data.loc[article_data.loc[:, "article_id"] == top_10_most_popular_article_ids[0], "heading"].item())
        contents.append(article_data.loc[article_data.loc[:, "article_id"] == top_10_most_popular_article_ids[0], "content"].item())

    recommendation_dict = {}
    
    for data in zip(article_ids, headings, contents):
        recommendation_dict[data[0]] = {
            "heading": data[1],
            "content": data[2]
        }

    return recommendation_dict