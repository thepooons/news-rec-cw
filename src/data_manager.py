import pandas as pd
from tqdm import tqdm
import yaml
import numpy as np
tqdm.pandas()

class DataManager(object):
    """Data Manager class"""
    def __init__(self, clickstream_data_path, article_vectors_data_path):
        self.data = pd.read_csv(clickstream_data_path) 
        self.article_vectors = pd.read_csv(article_vectors_data_path) 
        
    def merge_article_vectors(self):
        def get_vector(article_id, article_vectors, vector_columns):
            article_vector = article_vectors.loc[
                article_vectors.loc[:, "article_id"] == article_id,
                vector_columns
            ]
            return article_vector.values.reshape(1, -1)

        vector_columns = [*[f"heading_{i}" for i in range(50)], *[f"content_{i}" for i in range(50)]]
        article_vectors_ = self.data.loc[:, "article_id"].progress_apply(
            lambda article_id: get_vector(
                article_id=article_id, 
                article_vectors=self.article_vectors,
                vector_columns=vector_columns
            )
        )
        article_vectors_ = np.concatenate(article_vectors_)
        article_vectors_ = pd.DataFrame(
            data=article_vectors_,
            columns=vector_columns
        )

        self.data = pd.concat(objs=[self.data, article_vectors_], axis=1)

    def train_test_split(self, test_fraction):
        """does:
            1. split the data into train and test data
            2. return train and test data
        """
        train_data = pd.DataFrame(columns=self.data.columns)
        test_data  = pd.DataFrame(columns=self.data.columns)

        for user_data in tqdm(self.data.groupby("user_id"), desc="Train-Test Split"):
            num_sessions = len(user_data[1].groupby("session_id"))
            train_test_split_index = int(num_sessions * (1 - test_fraction)) # 🧠
            for session_data in user_data[1].groupby("session_id"):
                session_id = session_data[0]
                if session_id < train_test_split_index:
                    train_data = pd.concat([train_data, session_data[1]], axis=0)
                else:
                    test_data = pd.concat([test_data, session_data[1]], axis=0)
        train_data.to_csv("data/generated/train_clickstream.csv", index=False)
        test_data.to_csv("data/generated/test_clickstream.csv", index=False)

if __name__ == "__main__":
    # read the configuration
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dm = DataManager(
        clickstream_data_path=config["clickstream_data_path"],
        article_vectors_data_path=config["clustered_vectorized_data_path"]
    )

    dm.merge_article_vectors()
    dm.train_test_split(test_fraction=config["test_fraction"])