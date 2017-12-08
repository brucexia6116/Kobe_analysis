����pipline���������ݽ��д���
��ֵ�����ݲ��ñ�׼������
���������ݲ���one-hot����

����ָ���չʾ
���ݶ������չʾ
���ϵ����չʾ
�����չʾ  pandas-profiling
�Ʊ�Ͷ�������չʾ

�мල��Logistics�ع�(L1(Lasso)��L2)
	  ��������
	  �����ɭ��
	  ��GDBT
	  : Adaboost
	  ��XgBoost
	  ��svm
���η�ʽ��grid_search
��������Ҫ��չʾ
���������ս��
AUC��ͼ


ʹ��Logistics��
���Ų������
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
׼ȷ�ʣ� 0.614785992218
��ȷ�ʣ� 0.640486725664
�ٻ��ʣ� 0.332949971248
 F1  �� 0.438138479001
 AUC �� 0.589679144414
             precision    recall  f1-score   support
        0.0       0.61      0.85      0.71      4232
        1.0       0.64      0.33      0.44      3478
avg / total       0.62      0.61      0.59      7710

ʹ��svm��
���Ų������
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
׼ȷ�ʣ� 0.481841763943
��ȷ�ʣ� 0.365574622985
�ٻ��ʣ� 0.202127659574
 F1  �� 0.260322162562
 AUC �� 0.456923943209
             precision    recall  f1-score   support
        0.0       0.52      0.71      0.60      4232
        1.0       0.37      0.20      0.26      3478
avg / total       0.45      0.48      0.45      7710

ʹ�þ�������
���Ų������
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
׼ȷ�ʣ� 0.618287937743
��ȷ�ʣ� 0.656341320865
�ٻ��ʣ� 0.322886716504
 F1  �� 0.432838697244
 AUC �� 0.591972658819
             precision    recall  f1-score   support
        0.0       0.61      0.86      0.71      4232
        1.0       0.66      0.32      0.43      3478
avg / total       0.63      0.62      0.59      7710


ʹ��RandomForest:
���Ų������
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
׼ȷ�ʣ� 0.613748378729
��ȷ�ʣ� 0.64400921659
�ٻ��ʣ� 0.321449108683
 F1  �� 0.428845416187
 AUC �� 0.587709431468
             precision    recall  f1-score   support
        0.0       0.60      0.85      0.71      4232
        1.0       0.64      0.32      0.43      3478
avg / total       0.62      0.61      0.58      7710

ʹ��GBDT:
���Ų������
	learning_rate=0.1
	n_estimators=20
	max_depth=3
	min_samples_leaf=60
    min_samples_split =1000
	max_features='sqrt'
	subsample=0.8
	random_state=10
׼ȷ�ʣ� 0.611932555123
��ȷ�ʣ� 0.640625
�ٻ��ʣ� 0.318286371478
 F1  �� 0.425278524779
 AUC �� 0.585773620522
             precision    recall  f1-score   support
        0.0       0.60      0.85      0.71      4232
        1.0       0.64      0.32      0.43      3478
avg / total       0.62      0.61      0.58      7710

ʹ��Adaboost:
���Ų������
	algorithm:'SAMME.R'
	base_estimator:None
	learning_rate:0.31999999999999995
	n_estimators:80
	random_state:None
׼ȷ�ʣ� 0.615823605707
��ȷ�ʣ� 0.647935779817
�ٻ��ʣ� 0.324899367453
 F1  �� 0.432784373803
 AUC �� 0.589907150645
             precision    recall  f1-score   support
        0.0       0.61      0.85      0.71      4232
        1.0       0.65      0.32      0.43      3478
avg / total       0.63      0.62      0.58      7710

ʹ��XGBoost
���Ų������
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
׼ȷ�ʣ� 0.612451361868
��ȷ�ʣ� 0.635060639471
�ٻ��ʣ� 0.331224841863
 F1  �� 0.43537414966
 AUC �� 0.587398810346
             precision    recall  f1-score   support
        0.0       0.61      0.84      0.70      4232
        1.0       0.64      0.33      0.44      3478
avg / total       0.62      0.61      0.58      7710
