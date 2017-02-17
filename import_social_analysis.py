import numpy as np
import xgboost as xgb
import pickle

model = pickle.load(open("models/boosted-type-2-alpha-0.5-eta-0.05-subsample-1-max-depth-6-8-0.41893.p", "r"))

def interest_score(im):
    features = np.hstack((im['inception_pool'], im['faces']['num'], im['faces']['total'], im['faces']['largest']))
    feature_matrix = xgb.DMatrix(features.reshape(1,-1))
    score = float(model.predict(feature_matrix, ntree_limit=model.best_iteration)[0])

    return score