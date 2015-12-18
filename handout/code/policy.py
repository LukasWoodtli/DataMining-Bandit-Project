import numpy.random
import numpy as np
from numpy.linalg import inv
ALPHA = 0.26
DIMENSION = 6


class Article:
    def __init__(self, M=np.identity(DIMENSION),
                 w=np.zeros(DIMENSION), b=np.zeros(DIMENSION),
                 art_features=None):
        self.M = M
        self.M_inv = inv(M)
        self.w = w
        self.b = b
        self.art_features = art_features

AllArticles = {}

# Local score
# Random CTR: max 0.06
# LinUCB: CTR=0.066667


# Baseline Hard 	0.065825
# Baseline Easy 	0.044115


BestChoice = None
z_t = None

def set_articles(articles):
    # global AllArticles
    # for x in articles:
    #     features = articles[x]
    #     article = Article(art_features=features)
    #     AllArticles[x] = article
    pass

def update(reward):
    global AllArticles
    if reward == -1:
        return

    AllArticles[BestChoice].M = AllArticles[BestChoice].M + np.outer(z_t, z_t)
    AllArticles[BestChoice].M_inv = inv(AllArticles[BestChoice].M)
    AllArticles[BestChoice].b = AllArticles[BestChoice].b + np.multiply(z_t, reward)
    AllArticles[BestChoice].w = np.dot(AllArticles[BestChoice].M_inv, AllArticles[BestChoice].b)
    
    
def reccomend(time, user_features, articles):
    global BestChoice, z_t, AllArticles
    MaxUCB = 0   # -int.inf
    z_t = np.array(user_features)
    
    for art in articles:
        if not art in AllArticles:
            AllArticles[art] = Article()
        
        # Everything moved to update, where possible
        UCB = np.dot(AllArticles[art].w, z_t) + ALPHA * np.sqrt(np.dot(np.dot(z_t, AllArticles[art].M_inv), z_t))
        
        if UCB > MaxUCB:
            MaxUCB = UCB
            BestChoice = art
            
    return BestChoice
