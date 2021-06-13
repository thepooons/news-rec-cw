import pandas as pd

class TrainTestSplit(object):
    """train-test split class"""
    def __init__(self, clickstream_data_path):
        self.data = pd.read_csv(clickstream_data_path)

    def train_test_split(self, test_fraction):
        """does:
            1. split the data into train and test data
            2. return train and test data
        """
        train_data = pd.DataFrame(columns=self.data.columns)
        test_data  = pd.DataFrame(columns=self.data.columns)

        for user_data in self.data.groupby("user_id"):
            num_sessions = len(user_data[1].groupby("session_id"))
            train_test_split_index = int(num_sessions * (1 - test_fraction)) # 🧠
            for session_data in user_data[1].groupby("session_id"):
                session_id = session_data[0]
                if session_id < train_test_split_index:
                    train_data = pd.concat([train_data, session_data[1]], axis=0)
                else:
                    test_data = pd.concat([test_data, session_data[1]], axis=0)
        return train_data, test_data