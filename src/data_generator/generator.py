import pandas as pd
import numpy as np
import yaml
from itertools import combinations


class ClickStreamGen(object):
    def __init__(
        self,
        num_users,
        clickstream_data_path,
        clustered_article_data_path
    ):
        self.clickstream_data = pd.DataFrame(
            columns=["user_id", "session_id", "article_id",
                "rank", "click", "time_spent"]
        )
        self.num_users = num_users
        self.clickstream_data_path = clickstream_data_path
        self.clustered_article_data = pd.read_csv(clustered_article_data_path)

        # collaboration aspect
        cluster_article_prob_hash = {}
        for cluster_id in np.unique(self.clustered_article_data.loc[:, "cluster_id"]):
            cluster_articles = self.clustered_article_data.loc[
                self.clustered_article_data.loc[:, "cluster_id"] == cluster_id, "article_id"
            ].values
            article_prob_hash = {}
            for article_id in cluster_articles:
                article_prob_hash[article_id] = 1
            cluster_article_prob_hash[cluster_id] = article_prob_hash
        self.cluster_article_prob_hash = cluster_article_prob_hash

    def generate(self):
        # generate lists
        self.user_ids, self.session_ids, self.user_num_sessions_hash = self._generate_user_id_session_ids()
        self.article_ids = self._generate_article_ids()
        self.ranks = self._generate_rank()
        
        # test lists 
        lists_hash = {
            "user_ids": self.user_ids,
            "session_ids": self.session_ids,
            "article_ids": self.article_ids,
            "ranks": self.ranks,

        }
        for list1, list2 in combinations(lists_hash.keys(), 2):    
            assert len(lists_hash[list1]) == len(lists_hash[list2]), \
                f"length of {list1} ({len(lists_hash[list1])}) and {list2} ({len(lists_hash[list2])}) doesn't match"
        
        # save clickstream data
        print(len(self.article_ids), len(np.unique(self.article_ids)))

    def _generate_user_id_session_ids(self):
        """generate a list of session ids spanning `self.num_users`
        """
        user_ids = []
        session_ids = []
        user_num_sessions_hash = {}
        for user_id in range(self.num_users):
            user_num_sessions_hash[user_id] = 0
            num_sessions = 5 #TODO: sample this value from a distribution
            for session_id in range(1, num_sessions + 1):
                user_ids.extend([user_id] * 10)
                session_ids.extend([session_id] * 10) # 10 -> num articles recommended
                user_num_sessions_hash[user_id] += 1

        return user_ids, session_ids, user_num_sessions_hash

    def _generate_article_ids(self):
        """sample 10 articles for each user for each session from a cluster
        """
        session_size = 10 # number of articles per session
        article_ids = []
        for user_id in self.user_num_sessions_hash.keys():
            num_sessions = self.user_num_sessions_hash[user_id]
            user_preference_cluster_id = 3 #TODO: sample this from a uniform distribution

            user_preference_article_ids = ClickStreamGen.get_cluster_article_ids(
                cluster_article_prob_hash=self.cluster_article_prob_hash,
                user_preference_cluster_id=user_preference_cluster_id,
                num_sessions=num_sessions,
                session_size=session_size,
            )
            for article_id in list(user_preference_article_ids):
                self.cluster_article_prob_hash[user_preference_cluster_id][article_id] += 1
            article_ids.extend(user_preference_article_ids)
        return article_ids
            
    @staticmethod
    def get_cluster_article_ids(
        cluster_article_prob_hash,
        user_preference_cluster_id,
        num_sessions,
        session_size
    ):
        #NOTE: will break if num_sessions * session_size > len(cluster_article_prob_hash[\
        # user_preference_cluster_id])
        article_prob_hash = cluster_article_prob_hash[user_preference_cluster_id]
        random_nonuniform_gen = np.random.default_rng()
        user_preference_article_ids = random_nonuniform_gen.choice(
            replace=False,
            a=np.array(list(article_prob_hash.keys())),
            size=num_sessions*session_size,
            p=np.array(list(article_prob_hash.values()))/sum(article_prob_hash.values())
        )
        
        return list(user_preference_article_ids)

    def _generate_rank(self):
        """generate a list of range(1, 11) for each session for each user
        """
        num_sessions = sum(self.user_num_sessions_hash.values())
        ranks = []
        for _ in range(num_sessions):
            ranks.extend(list(range(1, 11)))
        return ranks

    def _generate_click_time_spent(self):
        """generate binary vector, real numbered vector of length 10 for each session for each user
        """
        pass



if __name__ == "__main__":
    # read the configuration
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_generator = ClickStreamGen(
        num_users=config["total_users"],
        clickstream_data_path=config["clickstream_data_path"],
        clustered_article_data_path=config["clustered_vectorized_data_path"],
    )

    data_generator.generate()