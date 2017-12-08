# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"] #解决plt乱码
plt.rcParams['axes.unicode_minus'] = False            #解决plt乱码
from sklearn.model_selection  import train_test_split
from sklearn.model_selection  import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,\
    precision_score, recall_score, accuracy_score,f1_score,roc_auc_score,\
    roc_curve,auc
#读入数据
data = pd.read_csv('Kobe_data.csv')
data.head()

#删除没有用的列
dorp = ['action_type','game_event_id','game_id','team_id','team_name','matchup','shot_id','season']
data = data.drop(dorp,axis = 1)

#缺失值检查
data.isnull().sum()#返回每列包含的缺失值的个数
data = data.dropna()

#对数据进行预处理
data['opponent'].unique()
data.columns.tolist()
#计算科比年龄1978
data['age'] = data['game_date'].apply(lambda x: int(x.split('-')[0]))-1978
data = data.drop('game_date',axis = 1)

data['period'] = data['period'].astype('object')
data['shot_made_flag'].value_counts()
data.columns
data.dtypes
data.shape

#变量的可视化
#科比投篮
plt.figure(figsize=(10,15))
a = data.groupby('shot_zone_basic')
b = ['silver','orange','r','gold','dodgerblue','brown','gold']
for a1, b1 in zip(a,b):
    plt.scatter(a1[1].loc_x,a1[1].loc_y,color = b1 ,alpha=0.1)
#变量探索
import seaborn as sns
sns.set( palette="muted", color_codes=True) #设置图风格
sns.distplot(data.shot_distance,bins = 20)

sns.barplot(x=data.shot_zone_area,y=data.shot_made_flag)
sns.barplot(x=data.shot_type,y=data.shot_made_flag)
sns.barplot(x=data.shot_made_flag,y= data.opponent)
sns.barplot(x=data.shot_made_flag,y= data.combined_shot_type)

sns.pointplot(x=data.age,y= data.shot_made_flag,hue = data.shot_type,
              markers =['^','o'],linestyle =['-','--'] )

sns.jointplot(data.age,data.shot_distance, kind='reg')

#对自变量进行独热编码
data2 = pd.get_dummies(data.iloc[:,0:17])
data2.columns.tolist()

#构造模型数据集
data3 = data2.drop(['shot_made_flag'],axis = 1)
data3['label'] = data2['shot_made_flag']

#进行训练集与测试集的划分
data3.shape
X , y = data3.iloc[:,0:75].values , data3.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.3,random_state=1)

#使用逻辑回归
from sklearn.linear_model import LogisticRegression
clf = GridSearchCV(estimator=LogisticRegression(),
                  param_grid=[{'C':(0.01,0.1,1,10),
                               'penalty':('l1','l2')}],
                  scoring='roc_auc',
                  n_jobs = -1,
                  cv = 5)

#使用SVM
from sklearn.svm import SVC
clf = GridSearchCV(estimator=SVC(max_iter = 50,probability = True),
                  param_grid=[{'C':[0.001,0.01,0.1,1,10],
							   'kernel':['rbf','sigmoid']}],
                  scoring='roc_auc',
                  n_jobs=2,
                  cv = 5)

#使用决策树算法
from sklearn import tree
clf = GridSearchCV(estimator=tree.DecisionTreeClassifier(),
                  param_grid=[{'criterion':('gini','entropy'),
	    					   'splitter':('best','random'),
							   'max_depth':range(3,8,1),
							   'min_samples_split':range(2,11,1),
							   'min_samples_leaf':range(2,11,1)}],
                  scoring='roc_auc',
                  n_jobs = -1,
                  cv = 5)

#使用随机森林
from sklearn.ensemble import RandomForestClassifier
clf = GridSearchCV(estimator=RandomForestClassifier(),
                  param_grid=[{'n_estimators':range(10,100,10),
	    					   'criterion':('gini','entropy'),
							   'max_depth':range(3,8,1),
							   'min_samples_split':range(2,11,1),
							   'min_samples_leaf':range(2,11,1)}],
                  scoring='roc_auc',
                  n_jobs = -1,
                  cv = 5)

#使用GBDT
from sklearn.ensemble import GradientBoostingClassifier
clf = GridSearchCV(estimator=GradientBoostingClassifier(),
                   param_grid=[{'learning_rate':np.linspace(0.1,1.2,6),
							   'n_estimators':range(10,100,10),
	    					   'criterion':('friedman_mse','mse','mae'),
							   'max_depth':range(3,8,1),
							   'min_samples_split':range(2,11,1),
							   'min_samples_leaf':range(2,11,1)}],
                   scoring='roc_auc',
                   n_jobs=-1,
                   cv = 5)

#使用Adaboost
from sklearn.ensemble import AdaBoostClassifier
clf = GridSearchCV(estimator=AdaBoostClassifier(),
                  param_grid=[{'learning_rate':np.linspace(0.1,1.2,6),
							   'n_estimators':range(10,100,10),
							   'algorithm':('SAMME','SAMME.R')}],
                  scoring='roc_auc',
                  n_jobs=-1,
                  cv = 5)
				  
#使用XGBoost
from xgboost import XGBClassifier
clf = GridSearchCV(estimator=XGBClassifier(),
                  param_grid=[{'max_depth':range(3,9,1),
							   'gamma':[0.1,0.2],
							   'subsample':np.linspace(0.5,1,5),
                               'colsample_bytree':np.linspace(0.5,1,5),
                               'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
                               'learning_rate':[0.01,0.1,1]}],
                  scoring='roc_auc',
                  n_jobs=2,
                  cv = 5)


clf.fit(X_train,y_train)
# 预测结果
y_pred_array = clf.predict(X_test)
y_pred_prob =clf.predict_proba(X_test)
print('最优参数组合')
best_parameters=clf.best_estimator_.get_params()
for param_name in sorted(best_parameters.keys()):
    print('\t%s:%r' %(param_name,best_parameters[param_name]))

print('准确率：',accuracy_score(y_test,y_pred_array))
print('精确率：',precision_score(y_test,y_pred_array))
print('召回率：',recall_score(y_test,y_pred_array))
print(' F1  ：',f1_score(y_test,y_pred_array))
print(' AUC ：',roc_auc_score(y_test,y_pred_array))
#模型效果
print(classification_report( y_true=y_test, y_pred=y_pred_array ))
#画出混淆矩阵图
confmat = confusion_matrix( y_true=y_test, y_pred=y_pred_array )
fig,ax = plt.subplots(figsize=(2.5,2.5))
ax.matshow(confmat,cmap=plt.cm.Blues,alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
plt.xlabel('预测值')
plt.ylabel('实际值')
plt.show()

#ROC曲线画图
fpr,tpr, _ = roc_curve(y_test, y_pred_prob[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()

###########最后的模型效果图##############
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(C=0.1,class_weight=None,dual=False,fit_intercept=True,
	                     intercept_scaling=1,max_iter=100,multi_class='ovr',
                         n_jobs=1,penalty='l1',random_state=None,solver='liblinear',
                         tol=0.0001,verbose=0,warm_start=False)
clf1.fit(X_train,y_train)
y1_pred_prob =clf1.predict_proba(X_test)

from sklearn.svm import SVC
clf2 = SVC(C=0.10,cache_size=200,class_weight=None,coef0=0.0,decision_function_shape='ovr',
           degree=3,gamma='auto',kernel='sigmoid',max_iter=50,probability=True,random_state=None,
		   shrinking=True,tol=0.001,verbose=False)
clf2.fit(X_train,y_train)
y2_pred_prob =clf2.predict_proba(X_test)		   

from sklearn import tree
clf3 = tree.DecisionTreeClassifier(class_weight=None,criterion='entropy',max_depth=6,max_features=None,
	                               max_leaf_nodes=None,min_impurity_decrease=0.0,min_impurity_split=None,
                                   min_samples_leaf=9,min_samples_split=3,min_weight_fraction_leaf=0.0,
	                               presort=False,random_state=None,splitter='random')
clf3.fit(X_train,y_train)
y3_pred_prob =clf3.predict_proba(X_test) 

from sklearn.ensemble import RandomForestClassifier
clf4 = RandomForestClassifier(bootstrap=True,class_weight=None,criterion='gini',max_depth=7,max_features='auto',
	                       max_leaf_nodes=None,min_impurity_decrease=0.0,min_impurity_split=None,min_samples_leaf=7,
	                       min_samples_split=8,min_weight_fraction_leaf=0.0,n_estimators=90,n_jobs=1,oob_score=False,
						   random_state=None,verbose=0,warm_start=False)
clf4.fit(X_train,y_train)
y4_pred_prob =clf4.predict_proba(X_test)   

from sklearn.ensemble import AdaBoostClassifier
clf5 = AdaBoostClassifier(algorithm='SAMME.R',base_estimator=None,learning_rate=0.32,n_estimators=80,random_state=None)
clf5.fit(X_train,y_train)
y5_pred_prob =clf5.predict_proba(X_test) 

from sklearn.ensemble import GradientBoostingClassifier
clf6 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=20,max_depth=3,min_samples_leaf=60,
                                  min_samples_split =1000,max_features='sqrt',subsample=0.8,random_state=10)   
clf6.fit(X_train,y_train)
y6_pred_prob =clf6.predict_proba(X_test)   

from xgboost import XGBClassifier,plot_importance

clf7 = XGBClassifier(base_score=0.5,booster='gbtree',colsample_bylevel=1,colsample_bytree=1,gamma=0.1,
	                 learning_rate=0.1,max_delta_step=0,max_depth=5,min_child_weight=1,missing=None,
					 n_estimators=100,n_jobs=1,nthread=None,objective='binary:logistic',random_state=0,
					 reg_alpha=0,reg_lambda=1,scale_pos_weight=1,seed=None,silent=True,subsample=1)
clf7.fit(X_train,y_train)
y7_pred_prob =clf7.predict_proba(X_test)

#ROC曲线画图
plt.figure()
list = [y1_pred_prob,y2_pred_prob,y3_pred_prob,y4_pred_prob,
        y5_pred_prob,y6_pred_prob,y7_pred_prob]
list2 = ['LR','SVM','DT','RF','Ada','GBDT','XGB']
list3 = ['k','r','y','c','g','b','m']
for i in [0,1,2,3,4,5,6]:
    fpr,tpr, _ = roc_curve(y_test, list[i][:,1])
    roc_auc = auc(fpr, tpr)

    lw = 2
    plt.plot(fpr, tpr, color=list3[i],
         lw=lw, label='%s ROC curve (area = %0.2f)' % (list2[i],roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ')
    plt.legend(loc="lower right")
    plt.show()

#变量重要性
import matplotlib.pyplot as plt
ax=plot_importance(clf7,max_num_features = 20)
plt.show()  #没有这句只有debug模式才会显示。。
help(plot_importance)