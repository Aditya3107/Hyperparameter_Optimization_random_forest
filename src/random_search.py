import pandas as pd 
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

#read the dataset
if __name__ == '__main__':
    df = pd.read_csv('input/train.csv')
    X = df.drop('price_range',axis = 1).values
    y = df.price_range.values

    classifier = ensemble.RandomForestClassifier(n_jobs= -1)
    param_grid = {
        'n_estimators':np.arange(100,1500,100),
        'criterion': ['gini','entropy'],
        'max_depth': np.arange(1,20)
    }
    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter= 10,
        scoring = "accuracy",
        n_jobs=1,
        verbose=10,
        cv= 5 #5 fold cross validation
    )

    model.fit(X,y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())
