import numpy as np
import pandas as pd


class Profiler(object):
    """
    Collects the user profile and returns it
    """

    def __init__(self, train_data):
        # Initialize the class variable
        self.train_data = train_data
        self.article_data = pd.read_csv("../input/bbc-toi-yahoo-news-statsfeatures/bbc_toi_yahoo_news_clustered_vectored.csv")
        self.all_ids = np.unique(self.train_data["article_id"].tolist() + self.test_data["article_id"].tolist()).tolist()

    def get_user_profile(self, user_id):
        # Collect the user train data
        curr_user_train_data = self.train_data[self.train_data["user_id"] == user_id]

        # Collect ids that user has clicked on and spent time > 0
        ids_clicked = curr_user_train_data[(curr_user_train_data["click"] == 1) & (curr_user_train_data["time_spent"] > 0)]["article_id"].values
        ids_not_clicked = curr_user_train_data[(curr_user_train_data["click"] == 0)]["article_id"].values

        # Collect the vector representation
        columns = ["content_" + str(i) for i in range(50)]
        article_vector_watched = self.article_data.set_index("article_id").iloc[ids_clicked][columns]
        article_vector_not_watched = self.article_data.set_index("article_id").iloc[ids_not_clicked][columns]

        # Make a dict
        dict_curr_user = {
            "u_train_data": curr_user_train_data,
            "is_possible": curr_user_train_data.shape[0],
            "average_vector_watched": article_vector_watched.values.sum(axis=0),
            "average_vector_not_watched": article_vector_not_watched.values.sum(axis=0),
            "id_clicked": ids_clicked,
            "id_not_clicked": ids_not_clicked,
        }

	# Return the data
        return self.all_ids, dict_curr_user
