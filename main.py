import os
import pandas as pd
import tensorflow as tf
from src.evaluate import Evaluate
from src.hybrid_model.model import MfHybridModel
from src.hybrid_model.train_hybrid_model import TrainHybridModel
from src.hybrid_model.infer import infer
from src.utils import create_logger
from tqdm import tqdm
import yaml
import warnings

warnings.simplefilter("ignore")


class GlobalWrapper(object):
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
        self.pretrained = config["pretrained"]
        self.batch_size = config["batch_size"]
        self.total_users = config["total_users"]
        self.total_articles = config["total_articles"]
        self.pretrained_weights_path = config["pretrained_weights_path"]
        self.logger = create_logger("kjkr-poons-news-recsys")

    def perform_all(
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
            num_user=self.total_user + 1,
            item_dim=50,  # restricted by GloVe Vectors
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
        heading_cols = ["heading_%d" % i for i in range(50)]
        all_cols = heading_cols + ["user_id", "time_spent", "click"]

        # Combine the df_ft
        self.mapper["id"] = self.mapper["id"] + 1
        article_heading = (
            self.mapper.set_index("id")
            .loc[articles_picked, heading_cols]
            .reset_index(drop=True)
        )
        df_ft = pd.concat([df_ft, article_heading], axis=1)

        # Use the case choosen
        if self.pretrained:
            print("Loading Pretrained model weights.....")
            # Check type of function
            if self.evaluate == False:
                # Load the pretrained weights
                model.load_weights(self.pretrained_weights_path + "model_hybrid_api.h5")

                # Fine tune the model
                obj = TrainHybridModel(
                    model_obj=model,
                    train_data=df_ft,
                    test_data=None,
                    task_type="api",
                    save_path=self.pretrained_weights_path,
                )
            else:
                # Load the pretrained weights
                model.load_weights(
                    self.pretrained_weights_path + "model_hybrid_evaluation.h5"
                )

                # Fine tune the model
                obj = TrainHybridModel(
                    model_obj=model,
                    train_data=df_ft,
                    test_data=self.test_data,
                    task_type="evaluation",
                    save_path=self.pretrained_weights_path,
                )

            path_model = obj.train_model(
                batch_size=self.batch_size, epochs=self.epochs, verbose=True
            )
        else:
            # Collect the new data
            all_data = pd.concat([self.train_data[all_cols], df_ft[all_cols]], axis=0)

            if self.evaluate == False:
                # Train a new model
                obj = TrainHybridModel(
                    model_obj=model,
                    train_data=all_data,
                    test_data=self.test_data,
                    task_type="api",
                    save_path=self.pretrained_weights_path,
                )
            else:
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
            for user in tqdm(range(1, self.total_user - 993 + 1), position=0):
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
            return None
        else:
            # Reverse map the articles ids
            heading_all = []
            content_all = []

            # Map
            for ids in tqdm(ids_to_recc[:top_many], smoothing=0.5, position=0):
                # Collect the heading and content
                curr = self.mapper[(self.mapper["id"]) == ids][["heading", "content"]]
                heading = curr["heading"].values[0]
                content = curr["content"].values[0]

                # Append to the list
                heading_all.append(str(heading))
                content_all.append(str(content))

            # Return as dict
            return {"heading": heading_all, "content": content_all}


if __name__ == "__main__":
    tmp = GlobalWrapper(config_path="config.yaml")
    tmp.perform_all()
