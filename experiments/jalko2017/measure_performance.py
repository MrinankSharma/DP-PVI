import numpy as np

def compute_log_likelihood(predictions, y):
    mod = np.array(y)
    mod[mod==1]=0
    ll = np.sum(np.log(-mod + y*predictions))
    return ll

def compute_prediction_accuracy(predictions, y):
    accuracy = np.sum(np.abs(2 * (predictions > 0.5) - 1 + y) > 0) / np.size(y)
    return accuracy