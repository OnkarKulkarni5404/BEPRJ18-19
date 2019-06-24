from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

from History_Bits import HB
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(200,n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1)
clf=HB()
parameter={}
parameter['Num_of_group'] = [1,2]
parameter['group_size'] = [1,2]
parameter['learn_rate'] = [0.0001,0.001,0.01,0.1,1,10,100,200]
clf = GridSearchCV(clf, parameter, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
clf = clf.fit(X, y)
clf_best = clf.best_estimator_
print(clf_best)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
Z = clf_best.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y,s=20, edgecolor='k')
plt.show()