import numpy.random
import numpy as np
from collections import  namedtuple
from numpy.linalg import inv
ALPHA = 0.5

DIMENSION = 6

M = {}
w = {}
b = {}
art_features = {}



BestChoice = None
z_t = None

def set_articles(articles):
    global M, w, b, art_features
    for x in articles:
        art = x[0]
        features = x[1:]
        M[art] = np.identity(DIMENSION)
        w[art] = np.zeros(DIMENSION)
        b[art] = np.zeros(DIMENSION)
        art_features[art] = features


def update(reward):
    global z_t, ArticleParams
    if reward == -1:
        return
    M[BestChoice] += np.dot(z_t.transpose(), z_t)
    b[BestChoice] += np.multiply(z_t, reward)


def reccomend(time, user_features, articles):
    #return numpy.random.choice(articles, size=1)
    global M, b, w, BestChoice, z_t
    MaxUCB = 0

    for art in articles:
        if not art in M:
            M[art] = np.identity(DIMENSION)
            b[art] = np.zeros(DIMENSION)
            w[art] = np.zeros(DIMENSION)
        z_t = np.array(user_features)
        UCB = np.inner(w[art].transpose(), z_t) + ALPHA * np.sqrt(np.dot(np.dot(z_t.transpose(), inv(M[art])), z_t))
        if UCB > MaxUCB:
            MaxUCB = UCB
            BestChoice = art
    return BestChoice
