import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle


def predict(arr):
    # Load the model
    with open('/Users/abdessalambenayyad/desktop/Weather_prediction/final_model.sav', 'rb') as f:
        model = pickle.load(f)
    classes ={0: 'rain', 1: 'snow', 2: 'sun', 3: 'drizzle', 4: 'fog'}
    # return prediction as well as class probabilities
    preds = model.predict_proba([arr])[0]
    return (classes[np.argmax(preds)], preds)