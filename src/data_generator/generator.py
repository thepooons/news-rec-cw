import numpy as np
import pandas as pd
import operator
import math
import os
from collections import defaultdict
from tqdm import tqdm
import yaml


class ClickStreamGen(object):
    """
    Creates the Click Stream Data
    """

    def __init__(self, total_art, total_user, num_clusters, cluster_data, overtime_pref=False, alpha=0.04):
        # Initialize the variables
        self.total_art = total_art
        self.total_user = total_user
        self.total_sess = None
        self.sample_count = None
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.cluster_data = cluster_data
        self.overtime_pref = overtime_pref

    def _generate_session_ids(self):
        # Generate the session ids
        total_session_samples = np.random.poisson(
            lam=10, size=(self.total_user))
        self.sample_count = total_session_samples

        # Placeholder for total sessions
        total_sess = 0

        # Create the data using for loop
        session_ids = []
        for sess_size in total_session_samples:
            for sess_num in range(1, sess_size + 1):
                total_sess += 1
                session_ids.extend([sess_num] * 10)

        # Global length
        self.total_sess = total_sess

        # Return the session ids
        return session_ids

    def _generate_article_ids(self):
        # Click data
        session_ids = self._generate_session_ids()

        # Placeholder
        article_ids = []
        article_click_data = []

        # Start loop
        for z in tqdm(
            range(self.total_user),
            total=self.total_user,
            desc="Generating `article_id`s | Overtime_pref: %s" % self.overtime_pref,
        ):
            # Initially original probabilites
            selection_prob = {
                i: (1 / (self.num_clusters)) for i in range(self.num_clusters)
            }

            # Start loop and collect valeus from cluster
            for curr_sess in range(self.sample_count[z]):
                # Generate the user click data
                rand_aspect = np.random.rand(1)[0]
                clicks_curr_user = np.random.binomial(
                    n=1, p=rand_aspect, size=10)
                article_click_data.extend(clicks_curr_user)

                # assign a cluster for the current user
                cluster_assi_curr = np.random.randint(
                    low=1, high=self.num_clusters, size=1)[0]

                # Handle dups
                list_ids_watched = []

                # Check if alpha based pref method
                if self.overtime_pref:

                    # Handle count allotment
                    num_values_to_collect = [0] * self.num_clusters
                    for i in range(self.num_clusters):
                        # Make a choice
                        num_values_to_collect[i] = int(
                            math.ceil(selection_prob[i] * 10))

                        # Check if value has exceeded
                        if sum(num_values_to_collect) == 10:
                            break

                        if sum(num_values_to_collect) > 10:
                            # Collect how many values have exceeded
                            offset = sum(num_values_to_collect) - 10

                            # Make a choice again
                            choice = np.argsort(num_values_to_collect).tolist()[::-1][
                                :offset
                            ]

                            # Reduce offset
                            for index in choice:
                                num_values_to_collect[index] -= 1

                    # Cleaning phase 2
                    step_two = 0
                    for i in range(len(num_values_to_collect)):
                        if num_values_to_collect[i] < 0:
                            step_two += 1
                            num_values_to_collect[i] = 0

                    if step_two > 0:
                        # Make a choice again
                        choice = np.argsort(num_values_to_collect).tolist()[
                            ::-1][:step_two]

                        # Reduce offset
                        for index in choice:
                            num_values_to_collect[index] -= 1

                    # Make the cluster value dict
                    cluster_mapper = []

                    # sample values from the data
                    current_ids = []

                    # Append
                    for i in range(self.num_clusters):
                        # Append the collected ids
                        current_ids.extend(
                            np.random.choice(
                                self.cluster_data[i], size=num_values_to_collect[i]
                            ).tolist()
                        )

                        # Append the cluster value to which it belongs
                        cluster_mapper.extend([i] * num_values_to_collect[i])

                    # Update user preferences using click data
                    cluster_clicks = {i: 0 for i in range(self.num_clusters)}
                    for i in range(len(current_ids)):
                        if clicks_curr_user[i] == 1:
                            cluster_clicks[cluster_mapper[i]] += 1

                    # Update the values
                    pref_cluster = max(cluster_clicks.items(), key=operator.itemgetter(1))[
                        0
                    ]
                    for k, v in selection_prob.items():
                        if k == pref_cluster:
                            selection_prob[k] += self.alpha
                        else:
                            selection_prob[k] -= self.alpha / \
                                (len(selection_prob) - 1)

                    # Append
                    article_ids.extend(current_ids)

                else:
                    # Handle repition
                    while True:
                        # Check repititon
                        counter = 0

                        # Assign all values from a single cluster
                        choice_curr = np.random.choice(
                            self.cluster_data[cluster_assi_curr], size=10).tolist()

                        # Check counter values
                        for i in choice_curr:
                            if choice_curr in article_ids:
                                counter += 1
                                if counter > 1:
                                    break

                        if counter <= 1:
                            break

                    article_ids.extend(choice_curr)

                    # Return
        return article_ids, session_ids, article_click_data

    def _generate_article_rank(self):
        # Craete a placeholder
        article_rank = [i for i in range(1, 11)] * self.total_sess

        # Return  the article ranks
        return article_rank

    def _generate_time_spent(self):
        # Fetch data
        article_ids, session_ids, article_click_data = self._generate_article_ids()

        # Placeholder
        time_spent_all = [0] * len(article_click_data)

        # Start the loop and vroom vroom

        for i in range(len(article_click_data)):
            # If clicked
            if article_click_data[i] == 1:
                random_lam = np.random.choice(
                    a=[5, 20, 50, 100, 150, 200, 300, 500],
                    size=1,
                    p=[0.33, 0.2, 0.1, 0.1, 0.08, 0.07, 0.06, 0.06],
                )
                time_spent_all[i] = np.random.poisson(
                    lam=random_lam, size=(1))[0]

        # Return
        return article_click_data, time_spent_all, article_ids, session_ids

    def _generate_user_ids(self):
        # Make the user ids
        user_ids = []

        # Start loop and make ids:
        for i in range(self.total_user):
            user_ids.extend([i + 1] * self.sample_count[i] * 10)

        # Return
        return user_ids

    def _make_dataframe(self):
        # Collect all the components
        (
            article_click_data,
            time_spent_all,
            article_ids,
            session_ids,
        ) = self._generate_time_spent()
        article_rank = self._generate_article_rank()
        user_ids = self._generate_user_ids()

        # Assert everything
        l_i = {
            "u_id": user_ids,
            "s_id": session_ids,
            "a_id": article_ids,
            "ars": article_rank,
            "acr": article_click_data,
            "ats": time_spent_all,
        }
        for name_one, col_one in l_i.items():
            for name_two, col_two in l_i.items():
                assert len(col_one) == len(
                    col_two
                ), "Error Shape Mismatch : %s | %d -- %s | %d" % (
                    name_one,
                    len(col_one),
                    name_two,
                    len(col_two),
                )

        # Create dataframe
        data_frame = pd.DataFrame(
            {
                "user_id": user_ids,
                "session_id": session_ids,
                "article_id": article_ids,
                "article_rank": article_rank,
                "click": article_click_data,
                "time_spent": time_spent_all,
            }
        )

        # Convert click to binary Y/N from 1, 0
        data_frame["click"] = data_frame["click"].astype(int)

        # Save the dataframe
        if not (os.path.exists("data/generated")):
            os.mkdir("data/generated")
        data_frame.to_csv("data/generated/clickstream.csv", index=False)

        # Return
        return data_frame


if __name__ == "__main__":
    # read the configuration
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load the data
    cluster_data = pd.read_csv(config["clustered_vectorized_data_path"])
    # Make a default dict to store dictionaries
    c_data = defaultdict(list)

    # Create a dict
    for index, data in cluster_data.iterrows():
        c_data[data["cluster_id"]].append(data["id"] + 1)

    # Make an instance and create data
    inst = ClickStreamGen(
        total_art=config["total_articles"],
        total_user=config["total_users"],
        num_clusters=len(c_data),
        cluster_data=c_data,
        overtime_pref=False,
        alpha=0.02,
    )

    data_content_collab = inst._make_dataframe()
