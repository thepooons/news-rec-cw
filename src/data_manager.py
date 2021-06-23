import pandas as pd
from tqdm import tqdm
import yaml
import time
import numpy as np
from collections import defaultdict
import pickle

tqdm.pandas()


class DataManager(object):
    """Data Manager class"""

    def __init__(self, clickstream_data_path, article_vectors_data_path):
        self.data = pd.read_csv(clickstream_data_path)
        self.article_vectors = pd.read_csv(article_vectors_data_path)
        self.train_data = None
        self.test_data = None

    def merge_article_vectors(self):
        """adds columns of new heading and content text GloVe vectors"""

        def get_vector(article_id, article_vectors, vector_columns):
            article_vector = article_vectors.loc[
                article_vectors.loc[:,
                                    "article_id"] == article_id, vector_columns
            ]
            return article_vector.values.reshape(1, -1)

        vector_columns = [
            *[f"heading_{i}" for i in range(100)],
            *[f"content_{i}" for i in range(100)],
        ]
        article_vectors_ = self.data.loc[:, "article_id"].progress_apply(
            lambda article_id: get_vector(
                article_id=article_id,
                article_vectors=self.article_vectors,
                vector_columns=vector_columns,
            )
        )
        article_vectors_ = np.concatenate(article_vectors_)
        article_vectors_ = pd.DataFrame(
            data=article_vectors_, columns=vector_columns)

        self.data = pd.concat(objs=[self.data, article_vectors_], axis=1)

    def train_test_split(self, test_fraction):
        """1. split the data into train and test data
           2. return train and test data


        Args:
            test_fraction (float): fraction of all the data which will
             be hold out for validating recommender system
        """
        train_data = pd.DataFrame(columns=self.data.columns)
        test_data = pd.DataFrame(columns=self.data.columns)

        for user_data in tqdm(self.data.groupby("user_id"), desc="Train-Test Split"):
            num_sessions = len(user_data[1].groupby("session_id"))
            train_test_split_index = int(
                num_sessions * (1 - test_fraction))  # 🧠
            for session_data in user_data[1].groupby("session_id"):
                session_id = session_data[0]
                if session_id < train_test_split_index:
                    self.train_data = pd.concat(
                        [self.train_data, session_data[1]], axis=0)
                else:
                    self.test_data = pd.concat(
                        [self.test_data, session_data[1]], axis=0)

        # Save the train and test data
        self.train_data.to_csv(config["train_data_path"], index=False)
        self.test_data.to_csv(config["test_data_path"], index=False)

    def create_user_hist(self):
        # Save the user dict
        total_user = np.unique(self.train_data["user_id"].values.tolist()
                               + self.test_data["user_id"].values.tolist()).tolist()

        # Init_dict
        dict_user = defaultdict(int)
        for user in total_user:
            # Collect total user sessionss
            collect_total_sessions = len(np.unique(
                self.train_data[self.train_data["user_id"] == user]["session_id"]))

            # Append to the dict
            dict_user[user] += collect_total_sessions

        # Save to directory
        with open(config["user_hist"], "wb") as f:
            pickle.dump(dict_user, f)

        # Make a file to save current embedding size
        with open(config["embed_size_local"], 'w') as f:
            f.write('%d' % len(total_user))


if __name__ == "__main__":
    # read the configuration
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dm = DataManager(
        clickstream_data_path=config["clickstream_data_path"],
        article_vectors_data_path=config["clustered_vectorized_data_path"],
    )

    dm.merge_article_vectors()
    dm.train_test_split(test_fraction=config["test_fraction"])
    dm.create_user_hist()
