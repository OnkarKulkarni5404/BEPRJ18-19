import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from History_Bits import HB
from sklearn.model_selection import cross_val_score, cross_val_predict,GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve,roc_auc_score,accuracy_score
from joblib import dump

#from sklearn.neighbors import LocalOutilerFactor


def preprocessing(dataset):
    remove_index = ['date', 'open', 'high', 'low', 'close', 'volume','Adj Close']
    df = dataset.drop(remove_index, axis=1)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float64)
    y = df['y'].copy()
    X = df.drop('y', axis=1)
    return X, y


def main(c_name):
    Train = pd.read_csv('Training.csv')
    Test = pd.read_csv('Holdout.csv')
    X_train, y_train = preprocessing(Train)
    X_test, y_test = preprocessing(Test)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    dump(scaler,c_name+'_scaler.joblib')
    X_test_scaled=scaler.transform(X_test)
    index = X_train.index
    columns = X_train.columns
    X_scaled = pd.DataFrame(data=X_scaled, index=index, columns=columns)
    index = X_test.index
    columns = X_test.columns
    X_test_scaled = pd.DataFrame(data=X_test_scaled, index=index, columns=columns)
    clf_hb = HB()
    parameter = {}
    parameter['Num_of_group'] = [i for i in range(1, X_scaled.shape[1] + 1)]
    parameter['group_size'] = [i for i in range(1, X_scaled.shape[1] + 1)]
    parameter['learn_rate'] = [0.0001,0.001,0.01,0.1,1,10,100,200]
    clf = GridSearchCV(clf_hb, parameter, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    clf = clf.fit(X_scaled, y_train)
    clf_best = clf.best_estimator_
    y_pred_train = cross_val_predict(clf_best, X_scaled, y_train, cv=5)
    conf_matrix = confusion_matrix(y_train, y_pred_train)
    accuracy = accuracy_score(y_train, y_pred_train)
    precision = precision_score(y_train, y_pred_train)
    recall = recall_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)
    auc = roc_auc_score(y_train, y_pred_train)
    print(conf_matrix)
    print("Accuracy = {}".format(accuracy))
    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("F1 Score = {}".format(f1))
    print("ROC AUC = {}".format(auc))
    y_test_pred=clf_best.predict(X_test_scaled)
    print("Testing accuracy= {}".format(accuracy_score(y_test,y_test_pred)))
    c_name+='.joblib'
    dump(clf_best,c_name)

if __name__ == '__main__':
    main()
