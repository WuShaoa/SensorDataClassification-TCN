import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit,KFold
#from sklearn.preprocessing import normalize



def data_process(path, n_splits=2):
    #tscv = TimeSeriesSplit(n_splits=n_splits)
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    df = pd.read_csv(path)

    X = np.array(df[df.columns[:-1]])
    y = np.array(df[df.columns[-1]])
    print(X.shape)
    #for train_index, test_index in tscv.split(X):
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print("TRAIN:", train_index, " ", train_index.shape, "TEST:", test_index, " ", test_index.shape)
        yield normalize(X_train),y_train,normalize(X_test),y_test

def normalize(v):
    return (v + np.abs(np.min(v,axis=0))) / (np.max(v,axis=0) - np.min(v,axis=0))
#print(normalize(np.array([[1,2],[4,5]])) @ normalize(np.array([[1,2],[4,5]])).T )