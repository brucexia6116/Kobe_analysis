采用pipline方法对数据进行处理
数值型数据采用标准化处理
连续型数据采用one-hot编码

数据指标的展示
数据多变量的展示
相关系数的展示
总体的展示  pandas-profiling
科比投篮区域的展示

有监督：Logistics回归(L1(Lasso)、L2)
	  ：决策树
	  ：随机森林
	  ：GDBT
	  : Adaboost
	  ：XgBoost
	  ：svm
调参方式：grid_search
参数的重要性展示
迭代结果的战术
AUC画图


使用Logistics：
最优参数组合
	C:0.1
	class_weight:None
	dual:False
	fit_intercept:True
	intercept_scaling:1
	max_iter:100
	multi_class:'ovr'
	n_jobs:1
	penalty:'l1'
	random_state:None
	solver:'liblinear'
	tol:0.0001
	verbose:0
	warm_start:False
准确率： 0.614785992218
精确率： 0.640486725664
召回率： 0.332949971248
 F1  ： 0.438138479001
 AUC ： 0.589679144414
             precision    recall  f1-score   support
        0.0       0.61      0.85      0.71      4232
        1.0       0.64      0.33      0.44      3478
avg / total       0.62      0.61      0.59      7710

使用svm：
最优参数组合
	C:0.10
	cache_size:200
	class_weight:None
	coef0:0.0
	decision_function_shape:'ovr'
	degree:3
	gamma:'auto'
	kernel:'sigmoid'
	max_iter:50
	probability:True
	random_state:None
	shrinking:True
	tol:0.001
	verbose:False
准确率： 0.481841763943
精确率： 0.365574622985
召回率： 0.202127659574
 F1  ： 0.260322162562
 AUC ： 0.456923943209
             precision    recall  f1-score   support
        0.0       0.52      0.71      0.60      4232
        1.0       0.37      0.20      0.26      3478
avg / total       0.45      0.48      0.45      7710

使用决策树：
最优参数组合
	class_weight:None
	criterion:'entropy'
	max_depth:6
	max_features:None
	max_leaf_nodes:None
	min_impurity_decrease:0.0
	min_impurity_split:None
	min_samples_leaf:9
	min_samples_split:3
	min_weight_fraction_leaf:0.0
	presort:False
	random_state:None
	splitter:'random'
准确率： 0.618287937743
精确率： 0.656341320865
召回率： 0.322886716504
 F1  ： 0.432838697244
 AUC ： 0.591972658819
             precision    recall  f1-score   support
        0.0       0.61      0.86      0.71      4232
        1.0       0.66      0.32      0.43      3478
avg / total       0.63      0.62      0.59      7710


使用RandomForest:
最优参数组合
	bootstrap:True
	class_weight:None
	criterion:'gini'
	max_depth:7
	max_features:'auto'
	max_leaf_nodes:None
	min_impurity_decrease:0.0
	min_impurity_split:None
	min_samples_leaf:7
	min_samples_split:8
	min_weight_fraction_leaf:0.0
	n_estimators:90
	n_jobs:1
	oob_score:False
	random_state:None
	verbose:0
	warm_start:False
准确率： 0.613748378729
精确率： 0.64400921659
召回率： 0.321449108683
 F1  ： 0.428845416187
 AUC ： 0.587709431468
             precision    recall  f1-score   support
        0.0       0.60      0.85      0.71      4232
        1.0       0.64      0.32      0.43      3478
avg / total       0.62      0.61      0.58      7710

使用GBDT:
最优参数组合
	learning_rate=0.1
	n_estimators=20
	max_depth=3
	min_samples_leaf=60
    min_samples_split =1000
	max_features='sqrt'
	subsample=0.8
	random_state=10
准确率： 0.611932555123
精确率： 0.640625
召回率： 0.318286371478
 F1  ： 0.425278524779
 AUC ： 0.585773620522
             precision    recall  f1-score   support
        0.0       0.60      0.85      0.71      4232
        1.0       0.64      0.32      0.43      3478
avg / total       0.62      0.61      0.58      7710

使用Adaboost:
最优参数组合
	algorithm:'SAMME.R'
	base_estimator:None
	learning_rate:0.31999999999999995
	n_estimators:80
	random_state:None
准确率： 0.615823605707
精确率： 0.647935779817
召回率： 0.324899367453
 F1  ： 0.432784373803
 AUC ： 0.589907150645
             precision    recall  f1-score   support
        0.0       0.61      0.85      0.71      4232
        1.0       0.65      0.32      0.43      3478
avg / total       0.63      0.62      0.58      7710

使用XGBoost
最优参数组合
	base_score:0.5
	booster:'gbtree'
	colsample_bylevel:1
	colsample_bytree:1
	gamma:0.1
	learning_rate:0.1
	max_delta_step:0
	max_depth:5
	min_child_weight:1
	missing:None
	n_estimators:100
	n_jobs:1
	nthread:None
	objective:'binary:logistic'
	random_state:0
	reg_alpha:0
	reg_lambda:1
	scale_pos_weight:1
	seed:None
	silent:True
	subsample:1
准确率： 0.612451361868
精确率： 0.635060639471
召回率： 0.331224841863
 F1  ： 0.43537414966
 AUC ： 0.587398810346
             precision    recall  f1-score   support
        0.0       0.61      0.84      0.70      4232
        1.0       0.64      0.33      0.44      3478
avg / total       0.62      0.61      0.58      7710
