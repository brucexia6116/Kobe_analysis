from sklearn.datasets import load_iris
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy',
	max_depth=7,
	max_features=None,
	max_leaf_nodes=None,
	min_impurity_decrease=0.0,
	min_impurity_split=None,
	min_samples_leaf=6,
	min_samples_split=6,
	min_weight_fraction_leaf=0.0,
	presort=False,
	random_state=None,
	splitter='random')
clf = clf.fit(X_train,y_train)
tree.export_graphviz(clf,out_file="tree.dot"  )