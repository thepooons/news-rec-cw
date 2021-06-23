import os
import pandas as pd
import numpy as np
import tensorflow as tf
from src.evaluate import Evaluate
from src.hybrid_model.model import MfHybridModel
from src.hybrid_model.train_hybrid_model import TrainHybridModel
from src.hybrid_model.infer import infer
from src.utils import create_logger
from tqdm import tqdm
import yaml
import warnings


class Evaluate_Global(object):
    """
    Perform all the things
    """

    def __init__(self, config_path):
        # Initialize class variables
        if not (os.path.exists("data")):
            os.mkdir("data")
        if not (os.path.exists("data/common")):
            os.mkdir("data/common")
        if not (os.path.exists("data/generated")):
            os.mkdir("data/generated")
        if not (os.path.exists("data/GloVe")):
            os.mkdir("data/GloVe")
        if not (os.path.exists("logs")):
            os.mkdir("logs")
        if not (os.path.exists("weights")):
            os.mkdir("weights")

        # Make the initalize variables
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.evaluate = config["evaluate"]
        self.train_data = pd.read_csv(config["train_data_path"])
        self.test_data = pd.read_csv(config["test_data_path"])
        self.mapper = pd.read_csv(config["clustered_vectorized_data_path"])
        self.total_user = config["total_users"]
        self.epochs = config["epochs"]
        self.user_dimensions = config["user_dimensions"]
        self.learning_rate = config["learning_rate"]
        self.comb_type = config["comb_type"]
        self.batch_size = config["batch_size"]
        self.total_users = config["total_users"]
        self.pretrained_weights_path = config["pretrained_weights_path"]
        self.logger = create_logger("kjkr-poons-news-recsys")

    def evaluate_all(
        self,
        user_id=10,
        articles_picked=[1, 2, 3, 4],
        time_spent=[0, 53, 223, 0],
        click=[0, 1, 1, 0],
        top_many=10,
    ):
        """
        Combines all the code into a single pipeline for providing evaluationa and
        serving API requests

        Args:
            user_id (int, optional): The user id in case of API predictions. Defaults to 10.

            articles_picked (list, optional): The article that are clicked by user during
                                              frontend. Defaults to [1, 2, 3, 4].

            time_spent (list, optional): The list capturing the time spent of articles
                                         selected. Defaults to [0, 53, 223, 0].

            click (list, optional): The session click on articles picked.
                                    Defaults to [0, 1, 1, 0].

            top_many (int, optional): The number of recommmendations to serve.
                                      Defaults to 10.

        Returns:
            dict: The heading and content dict containing the recommendation
        """
        # Create the model using train and test data
        model = MfHybridModel(
            num_user=len(np.unique(self.train_data["user_id"].values.tolist(
            ) + self.test_data["user_id"].values.tolist())) + 4,
            item_dim=100,  # restricted by GloVe Vectors
            comb_type=self.comb_type,
            embed_dim=self.user_dimensions,
            lr=self.learning_rate,
        ).get_model()

        # Create a seperate dataframe to fine tune or train from scratch
        df_ft = pd.DataFrame(
            {
                "user_id": [user_id] * len(articles_picked),
                "time_spent": time_spent,
                "click": click,
            }
        )

        # Collect needed columns
        content_cols = ["content_%d" % i for i in range(100)]
        all_cols = content_cols + ["user_id", "time_spent", "click"]

        # Combine the df_ft
        article_content = (
            self.mapper.set_index("article_id")
            .loc[articles_picked][content_cols]
            .reset_index(drop=True)
        )
        df_ft = pd.concat([df_ft, article_content], axis=1)

        # Collect the new data
        all_data = pd.concat(
            [self.train_data[all_cols], df_ft[all_cols]], axis=0)

        # Train a new model
        obj = TrainHybridModel(
            model_obj=model,
            train_data=all_data,
            test_data=self.test_data,
            task_type="evaluation",
            save_path=self.pretrained_weights_path,
        )

        path_model = obj.train_model(
            batch_size=self.batch_size, epochs=self.epochs, verbose=True
        )

        # Fetch the latest model
        model = path_model

        # Make predictions for the user and return
        df_sorted, ids_to_recc = infer(
            model=model,
            train_data=self.train_data,
            user_id=user_id,
            all_ids_data=self.mapper,
        )

        if self.evaluate:
            # Placeholder for dict
            dict_users = {}

            # Collect the user data
            for user in tqdm(range(1, self.total_users + 1), position=0):
                dict_users[user] = infer(
                    model=model,
                    train_data=self.train_data,
                    user_id=user,
                    all_ids_data=self.mapper,
                )
            evaluation = Evaluate(
                train_data=self.train_data,
                test_data=self.test_data,
                recommendation_lists=dict_users,
                logger=self.logger,
            )
            evaluation.generate_eval_report()

            # Return None
            return None


if __name__ == "__main__":
    tmp = Evaluate_Global(config_path="config.yaml")
    tmp.evaluate_all()
