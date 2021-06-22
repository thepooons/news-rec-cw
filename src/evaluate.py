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
        logger,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.recommendation_lists = recommendation_lists
        self.logger = logger

    def generate_eval_report(
        self,
    ):
        """does:
        1. for each user in `self.recommendation_lists`:
            1.1 pass recommendations list, train, test data for this user to calculate metrics
        2. return eval report with metrics averaged over all the users
        """
        eval_report = {
            "ARHR": [],
            "Precision@k": [],
            "Recall@k": [],
        }

        # 1. for each user in `self.recommendation_lists`:
        for user_id in tqdm(self.recommendation_lists.keys()):
            user_data_train = self.train_data.loc[
                self.train_data.loc[:, "user_id"] == user_id
            ].reset_index(drop=True)
            user_data_test = self.test_data.loc[
                self.test_data.loc[:, "user_id"] == user_id
            ].reset_index(drop=True)
            user_average_time_spent = np.mean(user_data_train.loc[:, "time_spent"])
            
            recommendation_list = self.recommendation_lists[user_id][0][
                "article_id"
            ].values

            # 1.1 pass recommendations list, train, test data for this user to calculate metrics
            user_eval_report = Evaluate.evaluate_user(
                recommendation_list=recommendation_list,
                user_data_train=user_data_train,
                user_data_test=user_data_test,
            )
            for metric in eval_report.keys():
                if user_eval_report[metric] is not None:
                    eval_report[metric].append(user_eval_report[metric])

        # 2. return eval report with metrics averaged over all the users
        for metric in eval_report.keys():
            eval_report[metric] = np.mean(eval_report[metric])
        self.logger.info(f"EVAL REPORT: {eval_report}")

    @staticmethod
    def evaluate_user(recommendation_list, user_data_train, user_data_test):
        """generates an evaluation report for one user"""
        arhr = Metrics.ARHR(
            recommendation_list=recommendation_list,
            user_data_train=user_data_train,
            user_data_test=user_data_test,
        )
        pak = Metrics.precision_at_k(
            recommendation_list=recommendation_list,
            user_data_train=user_data_train,
            user_data_test=user_data_test,
        )
        rak = Metrics.recall_at_k(
            recommendation_list=recommendation_list,
            user_data_train=user_data_train,
            user_data_test=user_data_test,
        )

        user_eval_report = {
            "ARHR": arhr,
            "Precision@k": pak,
            "Recall@k": rak,
        }
        return user_eval_report
