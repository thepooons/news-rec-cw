import numpy as np
import pandas as pd
import surprise


class CollaborativeModelTrain(object):
    """
    Trains two collaborative model and saves them
    """

    def __init__(self, dataset_obj, data):
        # Initialize the instance variables
        self.dataset_obj = dataset_obj
        self.data = data

    def _train_save_models(self):
        # Create the click data
        click_data = self.data.copy()[["user_id", "article_id", "click"]]
        click_data.columns = ["user_id", "item_id", "rating"]

        # Create the time data
        time_data = self.data.copy()[["user_id", "article_id", "time_spent"]]

        # Scale the data
        max_time = np.max(time_data["time_spent"])
        min_time = np.min(time_data["time_spent"])
        time_data["time_spent"] = (
            time_data["time_spent"] - min_time) / (max_time - min_time)

        # Rename the columns
        time_data.columns = ["user_id", "item_id", "rating"]

        # Create the train_dataset
        dataset_click = self.dataset_obj.get_surprise_dataset(
            data=click_data, scale=(0, 1))
        dataset_time = self.dataset_obj.get_surprise_dataset(
            data=time_data, scale=(0, 1))

        # Train the models
        model_click_rate = surprise.prediction_algorithms.matrix_factorization.SVD()
        model_time_spent = surprise.prediction_algorithms.matrix_factorization.SVD()

        # Fit the model
        model_click_rate.fit(dataset_click)
        model_time_spent.fit(dataset_time)

        # Save the models
        surprise.dump.dump(file_name="model_click_rate.pkl",
                           algo=model_click_rate)
        surprise.dump.dump(file_name="model_time_spent.pkl",
                           algo=model_time_spent)

        return model_click_rate, model_time_spent, max_time, min_time
