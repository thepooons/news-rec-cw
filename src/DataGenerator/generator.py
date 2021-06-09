import numpy as np
import pandas as pd
import os

class ClickStreamGen(object):
    """
    Creates the Click Stream Data
    """

    def __init__(self, total_art, total_user):
        # Initialize the variables
        self.total_art = total_art
        self.total_user = total_user
        self.total_sess = None
        self.sample_count = None

    def _generate_session_ids(self):
        # Generate the session ids
        total_session_samples = np.random.poisson(
            lam=3, size=(self.total_user))
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
        # Random sample the articles
        article_ids = np.random.randint(
            low=1, high=self.total_art, size=self.total_sess * 10)

        # Return
        return article_ids

    def _generate_article_rank(self):
        # Craete a placeholder
        article_rank = [i for i in range(1, 11)] * self.total_sess

        # Return  the article ranks
        return article_rank

    def _generate_click_data(self):
        # Create the placeholder
        article_click_data = []

        # Possion sample before
        total_click_data = np.random.poisson(lam=4, size=self.total_user)

        # Start the loop and vroom vroom
        for i in range(self.total_user):
            for _ in range(self.sample_count[i]):
                rand_aspect = np.random.rand(1)[0]
                clicks_curr_user = np.random.binomial(
                    n=1, p=rand_aspect, size=10)
                article_click_data.extend(clicks_curr_user)
                
        # Assert
        assert len(article_click_data) == self.total_sess * 10, \
            "Error Click Data Shape Mismatch %d -- %d" % (len(article_click_data),
                                                          self.total_sess * 10)

        # Return the data
        return article_click_data

    def _generate_time_spent(self):
        # Fetch data
        article_click_data = self._generate_click_data()

        # Placeholder
        time_spent_all = [0] * len(article_click_data)

        # Start the loop and vroom vroom

        for i in range(len(article_click_data)):
            # If clicked
            if article_click_data[i] == 1:
                random_lam = np.random.choice(
                    a=[5, 20, 50, 100, 150, 200, 300, 500],
                    size=1,
                    p=[0.33, 0.2, 0.1, 0.1, 0.08, 0.07, 0.06, 0.06]
                )
                time_spent_all[i] = np.random.poisson(lam=random_lam, size=(1))[0]

        # Return
        return article_click_data, time_spent_all

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
        session_ids = self._generate_session_ids()
        article_ids = self._generate_article_ids()
        article_rank = self._generate_article_rank()
        article_click_rate, article_time_spent = self._generate_time_spent()
        user_ids = self._generate_user_ids()

        # Assert everything
        l_i = {
            "u_id": user_ids,
            "s_id": session_ids,
            "a_id": article_ids,
            "ars": article_rank,
            "acr": article_click_rate,
            "ats": article_time_spent
        }
        for name_one, col_one in l_i.items():
            for name_two, col_two in l_i.items():
                assert len(col_one) == len(col_two), \
                    "Error Shape Mismatch : %s | %d -- %s | %d" % (name_one,
                                                                   len(
                                                                   col_one),
                                                                   name_two,
                                                                   len(col_two))

        # Create dataframe
        data_frame = pd.DataFrame({
            "user_id": user_ids,
            "session_id": session_ids,
            "article_id": article_ids,
            "article_rank": article_rank,
            "click": article_click_rate,
            "time_spent": article_time_spent
        })

        # Convert click to binary Y/N from 1, 0
        data_frame["click"] = data_frame["click"].astype(int)

        # Save the dataframe
        if not(os.path.exists("data/generated")):
            os.mkdir("data/generated")
        data_frame.to_csv("data/generated/ClickDataPhaseI.csv", index=False)

        # Return
        return data_frame

if __name__ == "__main__":
    # Make an instance and create data
    TOTAL_USER = 1000
    TOTAL_ARTICLES = 150000
    inst = ClickStreamGen(total_art=TOTAL_ARTICLES, total_user=TOTAL_USER)
    inst._make_dataframe()