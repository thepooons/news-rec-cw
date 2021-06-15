from src.metrics import Metrics
from src.utils import create_logger
import numpy as np
import pandas as pd
from tqdm import tqdm

class Evaluate(object):
    """evaluation class"""
    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        recommendation_lists: dict,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.recommendation_lists = recommendation_lists
        self.logger = create_logger(self.__class__.__name__)

    def generate_eval_report(self,):
        """does:
            1. for each user in `self.recommendation_lists`:
                1.1 distinguish positive and negative user-item interactions
                1.2 log metrics
            2. return eval report with metrics averaged over all the users
        """
        eval_report = {
            "ARHR": [],
            "Precision@k": [],
            "Recall@k": [],
        }

        # 1. for each user in `self.recommendation_lists`:
        for user_id in tqdm(self.recommendation_lists.keys()):
            user_data = self.train_data.loc[self.train_data.loc[:, "user_id"] == user_id]
            user_average_time_spent = np.mean(user_data.loc[:, "time_spent"])

            # 1.1 distinguish positive and negative user-item interactions
            negative_item_list = user_data.loc[user_data.loc[:, "time_spent"] < user_average_time_spent, "article_id"]
            positive_item_list = user_data.loc[user_data.loc[:, "time_spent"] >= user_average_time_spent, "article_id"]
            recommendation_list = self.recommendation_lists[user_id][0]["article_id"].values
            
            # 1.2 log metrics
            user_eval_report = Evaluate.evaluate_user(
                recommendation_list=recommendation_list,
                positive_item_list=positive_item_list,
                negative_item_list=negative_item_list,
            )
            for metric in eval_report.keys():
                eval_report[metric].append(user_eval_report[metric])

        # 2. return eval report with metrics averaged over all the users
        for metric in eval_report.keys():
            eval_report[metric] = np.mean(eval_report[metric])
        self.logger.info(f"EVAL REPORT: {eval_report}")

    @staticmethod
    def evaluate_user(
        recommendation_list,
        positive_item_list,
        negative_item_list
    ):
        user_eval_report = {
            "ARHR": Metrics.ARHR(
                recommendation_list=recommendation_list,
                positive_item_list=positive_item_list,
                negative_item_list=negative_item_list
            ),
            "Precision@k": Metrics.precision_at_k(
                recommendation_list=recommendation_list,
                positive_item_list=positive_item_list,
                negative_item_list=negative_item_list
            ),
            "Recall@k": Metrics.recall_at_k(
                recommendation_list=recommendation_list,
                positive_item_list=positive_item_list,
                negative_item_list=negative_item_list
            )
        }
        return user_eval_report