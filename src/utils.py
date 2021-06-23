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