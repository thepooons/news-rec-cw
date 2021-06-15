import os
import numpy as np
import pandas as pd
import tensorflow as tf
from data_manager import DataManager
from evaluate import Evaluate
from metrics import Metrics
from CollaborativeModel.model import MfHybridModel
from CollaborativeModel.train_hybrid_model import TrainHybridModel
from CollaborativeModel.infer import infer
from DataGenerator.generator import ClickStreamGen
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.simplefilter("ignore")


class GlobalWrapper(object):
    """
    Perform all the things in this Repo
    """

    def __init__(self, evaluate=False):
        # Make the initalize variables
        # self.initialize_data()
        self.train_data = pd.read_csv("train_clickstream.csv")
        self.test_data = pd.read_csv("test_clickstream.csv")
        self.mapper = pd.read_csv("bbc_toi_yahoo_news_clustered_vectored.csv")
        self.evaluate = evaluate
        self.total_user = 1000
        self.epochs = 10

    def initialize_data(self):
        ####################################
        ### POONS TASK #####################
        ####################################

        # GENERATE ALL THE DATA AND POPULATE INIT PLACEHOLDERS ###
        pass

    def perform_all(self, user_id, articles_picked, time_spent, click, top_many=10, pretrained=True):
        # Create the model using train and test data
        model = MfHybridModel(num_user=1001, item_dim=50,
                              comb_type="concat", embed_dim=100, lr=0.0001).get_model()

        # Create a seperate dataframe to fine tune or train from scratch
        df_ft = pd.DataFrame({
            "user_id": [user_id] * len(articles_picked),
            "time_spent": time_spent,
            "click": click
        })

        # Collect needed columns
        heading_cols = ["heading_%d" % i for i in range(50)]
        all_cols = heading_cols + ["user_id", "time_spent", "click"]

        # Combine the df_ft
        self.mapper["id"] = self.mapper["id"] + 1
        df_ft = pd.concat([df_ft, self.mapper.set_index(
            "id").loc[articles_picked][heading_cols].reset_index(drop=True)], axis=1)

        # Use the case choosen
        if pretrained:
            # Load the pretrained weights
            model.load_weights("model_hybrid_api.h5")

            if self.evaluate == False:
                # Fine tune the model
                obj = TrainHybridModel(model_obj=model, train_data=df_ft,
                                       test_data=None, task_type="api")
            else:
                # Fine tune the model
                obj = TrainHybridModel(model_obj=model, train_data=df_ft,
                                       test_data=self.test_data, task_type="evaluation")

            path_model = obj.train_model(
                batch_size=128, epochs=self.epochs, verbose=True)
        else:
            # Collect the new data
            all_data = pd.concat(
                [self.train_data[all_cols], df_ft[all_cols]], axis=0)

            if self.evaluate == False:
                # Train a new model
                obj = TrainHybridModel(model_obj=model, train_data=all_data,
                                       test_data=self.test_data, task_type="api")
            else:
                # Train a new model
                obj = TrainHybridModel(model_obj=model, train_data=all_data,
                                       test_data=self.test_data, task_type="evaluation")

            path_model = obj.train_model(
                batch_size=128, epochs=self.epochs, verbose=True)

        # Fetch the latest model
        model = path_model

        # Make predictions for the user and return
        df_sorted, ids_to_recc = infer(model=model, train_data=self.train_data,
                                       user_id=user_id, all_ids_data=self.mapper)

        if self.evaluate:
            # Placeholder for dict
            dict_users = {}

            # Collect the user data
            for user in tqdm(range(1, self.total_user + 1), position=0):
                dict_users[user] = infer(model=model, train_data=self.train_data,
                                         user_id=user, all_ids_data=self.mapper)

            #### IMPLEMENT THE EVALUATE FUNCTIONS #################################
            #### POONS TASK YOU ARE GIVE THE TRAIN DATA ###########################
            #### DF_SORTED WILL GIVE THE SCORES AND ARTICLE IDS IN SORTED MANNER###
            #######################################################################
            return None
        else:
            # Reverse map the articles ids
            heading_all = []
            content_all = []

            # Map
            for ids in tqdm(ids_to_recc[:top_many], smoothing=0.5, position=0):
                # Collect the heading and content
                curr = self.mapper[(self.mapper["id"]) == ids][[
                    "heading", "content"]]
                heading = curr["heading"].values[0]
                content = curr["content"].values[0]

                # Append to the list
                heading_all.append(str(heading))
                content_all.append(str(content))

            # Return as dict
            return {
                "heading": heading_all,
                "content": content_all
            }


tmp = GlobalWrapper(evaluate=True)
tmp.perform_all(user_id=10, articles_picked=[
    1, 2, 3, 4], time_spent=[1, 2, 3, 4], click=[1, 1, 1, 1], pretrained=False)
