import numpy.random
import numpy as np
from numpy.linalg import inv
ALPHA = 0.5

DIMENSION = 6


class Article:
    def __init__(self, M=np.identity(DIMENSION),
                 w=np.zeros(DIMENSION), b=np.zeros(DIMENSION),
                 art_features=None):
        self.M = M
        self.w = w
        self.b = b
        self.art_features = art_features

AllArticles = {}


# Random CTR: max 0.06
# Baseline Hard 	0.065825
# Baseline Easy 	0.044115


BestChoice = None
z_t = None

def set_articles(articles):
    global AllArticles
    for x in articles:
        art = x[0]
        features = x[1:]
        article = Article(art_features=features)
        AllArticles[art] = article

def update(reward):
    global AllArticles
    art = AllArticles[BestChoice]
    art.M = art.M + np.dot(z_t.transpose(), z_t)
    art.b = art.b + np.multiply(z_t, reward)
    AllArticles[BestChoice] = art

def reccomend(time, user_features, articles):
    #return numpy.random.choice(articles, size=1)
    global BestChoice, z_t, AllArticles
    MaxUCB = 0

    for art in articles:
        if not art in AllArticles:
            AllArticles[art] = Article()
        z_t = np.array(user_features)
        w = AllArticles[art].w
        M = AllArticles[art].M
        UCB = np.inner(w.transpose(), z_t) + ALPHA * np.sqrt(np.dot(np.dot(z_t.transpose(), inv(M)), z_t))
        if UCB > MaxUCB:
            MaxUCB = UCB
            BestChoice = art
    return BestChoice
