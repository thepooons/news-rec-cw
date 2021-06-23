import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from src.hybrid_model.model import MfHybridModel
from src.hybrid_model.train_hybrid_model import TrainHybridModel
from src.hybrid_model.infer import infer
from src.utils import create_logger
from tqdm import tqdm
import yaml
from src.utils import top_10_recommendations


class APIRecommender(object):
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
        self.pretrained_weights_path = config["pretrained_weights_path"]
        self.logger = create_logger("kjkr-poons-news-recsys")
        self.user_hist_path = config["user_hist"]
        self.embed_local_path = config["embed_size_local"]

        self.top_10_recommendation_dict = top_10_recommendations(
                clickstream_data=self.train_data,
                article_data=self.mapper,
            )

    def load_existing_weight(self, model_old, embed_size=100, total_user=100):
        # Fetch the weights from old model
        weights = model_old.layers[2].get_weights()[0]
        layer_collected = []
        for layer in model_old.layers:
            if isinstance(layer, tf.keras.layers.Embedding):
                layer_collected.append(layer)

        # Make a new placeholder
        layer_weight_dict = {}
        for layer in range(len(layer_collected)):
            # Make a placeholder
            data_new = np.zeros((total_user + 1, embed_size))

            # Fill the weights
            for i in range(layer_collected[layer].weights[0].shape[0]):
                data_new[i] = weights[i]

            layer_weight_dict[layer] = data_new

        # Load the weights to a new matrix
        return layer_weight_dict

    def make_recommendations(
        self,
        user_id=102,
        article_id=[1, 2, 3, 4],
        time_spent=[0, 53, 223, 0],
        click=[0, 1, 1, 0],
        top_many=10,
    ):
        """
        Combines all the code into a single pipeline for providing evaluation and
        serving API requests

        Args:
            user_id (int, optional): The user id in case of API predictions. Defaults to 10.

            article_id (list, optional): The article that are clicked by user during
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
        # Load the pickle file and check if user is new or old
        with open(self.user_hist_path, "rb") as f:
            user_hist = pickle.load(f)

        with open(self.embed_local_path, "r") as f:
            size_embed = int(f.read().split()[0])

        # Check if user is old or not
        if user_id in user_hist.keys():
            print("Hello Old User.....")
            old_user = True
            show_trending = False
            user_hist[user_id] += 1
        else:
            print("Hello New User....")
            old_user = False
            show_trending = True
            user_hist[user_id] = 1
            # Load the embed size
            with open(self.embed_local_path, "w") as f:
                f.write('%d' % (len(user_hist) + 1))

        # Save again
        with open(self.user_hist_path, "wb") as f:
            pickle.dump(user_hist, f)

        ###################### USER COLLABORATIVE ASPECT ###################
        # Check if new user
        if show_trending and old_user == False:
            return self.top_10_recommendation_dict

        # Create a seperate dataframe to fine tune or train from scratch
        df_ft = pd.DataFrame(
            {
                "user_id": [user_id - 1] * len(article_id),
                "time_spent": time_spent,
                "click": click,
            }
        )

        # Collect needed columns
        content_cols = ["content_%d" % i for i in range(100)]
        all_cols = content_cols + ["user_id", "time_spent", "click"]

        # Combine the df_ft
        article_content = (
            self.mapper.set_index(
                "article_id").loc[article_id][content_cols].reset_index(drop=True)
        )
        df_ft = pd.concat([df_ft, article_content], axis=1)

        # Use the case choosen
        if self.pretrained and old_user == True:
            # Print
            print("Loading Pretrained model weights.....")

            # Load the pretrained weights
            weights_old = self.load_existing_weight(
                model_old=tf.keras.models.load_model(
                    self.pretrained_weights_path + "model_hybrid_api.h5"),
                total_user=size_embed - 1,
            )

            model = MfHybridModel(
                num_user=size_embed,
                item_dim=100,  # restricted by GloVe Vectors
                comb_type=self.comb_type,
                embed_dim=self.user_dimensions,
                lr=self.learning_rate,
                user_pretrained=weights_old
            ).get_model()

            # Fine tune the model
            obj = TrainHybridModel(
                model_obj=model,
                train_data=df_ft,
                test_data=None,
                task_type="api",
                save_path=self.pretrained_weights_path,
            )

            path_model = obj.train_model(
                batch_size=self.batch_size, epochs=self.epochs, verbose=True
            )
        else:
            # laod model
            model = MfHybridModel(
                num_user=size_embed,
                item_dim=100,  # restricted by GloVe Vectors
                comb_type=self.comb_type,
                embed_dim=self.user_dimensions,
                lr=self.learning_rate,
                user_pretrained=None
            ).get_model()

            # Collect the new data
            all_data = pd.concat(
                [self.train_data[all_cols], df_ft[all_cols]], axis=0)

            # Train a new model
            obj = TrainHybridModel(
                model_obj=model,
                train_data=all_data,
                test_data=self.test_data,
                task_type="api",
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

        # Reverse map the articles ids
        heading_all = []
        content_all = []
        article_id_all = []

        # Map
        for ids in tqdm(ids_to_recc[:top_many], position=0):
            # Collect the heading and content
            curr = self.mapper[(self.mapper["article_id"]) == ids][[
                "heading", "content"]]
            heading = curr["heading"].values[0]
            content = curr["content"].values[0]
            article_id = curr["article"].values[0]

            # Append to the list
            heading_all.append(str(heading))
            content_all.append(str(content))
            article_id_all.append(int(article_id))
        
        recommendation_dict = {}
        for index, data in enumerate(zip(article_id_all, heading_all, content_all)):
            recommendation_dict[index] = {
                "article_id": data[0],
                "heading": data[1],
                "content": data[2]
            }
            
        # Return as dict
        return recommendation_dict