# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:35:12 2018

@author: jomr
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier#KNN模型
from sklearn.tree import DecisionTreeClassifier#决策树
from sklearn.ensemble import (ExtraTreesClassifier,RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn import metrics
from sklearn.neural_network import MLPClassifier#多层感知器分类器
from sklearn.naive_bayes import GaussianNB#朴素贝叶斯https://scikit-learn.org/stable/modules/naive_bayes.html
from sklearn.linear_model import LogisticRegression#Logistic回归
#from sklearn.ensemble import RandomForestRegressor#随机森林
from sklearn.svm import SVC#支持向量机


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis#LDA降维
import pandas as pd
def get_stacking(clf, x_train, y_train, x_test, n_folds=10):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    """
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]

        clf.fit(x_tra, y_tra)
        second_level_train_set[test_index] = clf.predict(x_tst)
        test_nfolds_sets[:,i] = clf.predict(x_test)
    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set
def getDataSet(file):
    '''
    获取数据
    file是数据文件所在的绝对路径
    返回Dataframe对象
    '''
    return pd.read_csv(file,sep=";")

def getFeatures(X,Y):
    '''
    树算法可以计算特征的信息量
    '''
    model = ExtraTreesClassifier()
    model.fit(X,Y)
    print(model.feature_importances_)
def standard_data(data,qq,k,column_list):
    '''
    标准化、归一化数据
    '''
    data.drop(["id"],axis=1)#删除id列
    data[["QQ"]]=data["QQ"].map(lambda x:"1" if(x==qq) else "0")#分类标签，目标qq号为我们的1,其他为0
    for i in column_list:
        MAX=np.max(data[i])
        MIN=np.min(data[i])
        data[i]=data[i].map(lambda x:(x-MIN)/(MAX-MIN))
    #随机选取50行目标数据和假冒者数据
    data=sample_data(data,k)
    return data
def sample_data(data,k):
    '''
    从数据源中获取指定k条分类目标为1的数据与k条分类目标为0的数据
    '''
    #当数据量多于k时使用欠抽样,当数据量少于k时使用过抽样
    if(data[data['QQ']=='1'].shape[0]>k):
        positive_data=data[data["QQ"]=="1"].sample(n=k,axis=0)
    else:
        positive_data=data[data['QQ']=='1'].append(data[data["QQ"]=="1"].sample(n=k-data[data['QQ']=='1'].shape[0],replace=True,axis=0))
    if(data[data['QQ']=='0'].shape[0]>k):
        nagative_data=data[data["QQ"]=="0"].sample(n=k,axis=0)
    else:
        nagative_data=data[data['QQ']=='0'].append(data[data["QQ"]=="0"].sample(n=k-data[data['QQ']=='0'].shape[0],replace=True,axis=0))
    return positive_data.append(nagative_data)
def model_output(model_name,expected,predicted):
    print(model_name)
    # summarize the fit of the model
    print('RESULT')
    print(metrics.classification_report(expected, predicted))
    print('CONFUSION MATRIX')
    print(metrics.confusion_matrix(expected, predicted))

users=['1904119914@qq.com','1689105594@qq.com','642415813@qq.com',
       '784688321@qq.com','1359448166@qq.com']
df=getDataSet("mail_table.csv")#加载数据
#df=Standard_data(df)#对数据进行处理

#对区间型变量进行标准化归一化"finger_slides","touch_times","read_duration","average_touch_duration","mail_height",
#"mail_width","average_location_x","average_location_y","average_sliding_distance",
#"average_finger_pressure","average_sensor_x","average_sensor_y","average_sensor_z"
#第二个参数代表认证目标，，第三个参数代表每一类的数据量
df=standard_data(df,users[3],150,["finger_slides","touch_times","read_duration","average_touch_duration","mail_height",
                     "mail_width","average_location_x","average_location_y","average_sliding_distance",
                     "average_finger_pressure","average_sensor_x","average_sensor_y","average_sensor_z"])

X=df.drop(["QQ"],axis=1).values#除QQ一列都是自变量
Y=df.QQ.values#QQ为自变量


##不降维
#X_new=X

##PCA降维
#pca=PCA(n_components='mle')
#pca.fit(X)
#X_new=pca.transform(X)

#LDA降维
lda = LinearDiscriminantAnalysis(solver="eigen",n_components=1)
lda.fit(X,Y)
X_new = lda.transform(X)


#数据拆分为训练集和测试集
train_x, test_x, train_y, test_y =train_test_split(X_new,Y,test_size=0.3,random_state=0,stratify=Y)
#x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
train_sets = []
test_sets = []

#基分类器model1~5
model1=LogisticRegression()
model2=GaussianNB()
model3=KNeighborsClassifier()
model4=SVC()

#次级分类器dt_model、model6~9
dt_model = DecisionTreeClassifier()
model5=DecisionTreeClassifier(criterion='entropy')#使用信息熵作为划分标准，对决策树进行训练
model6= MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50, 50), random_state=1)
model7 = RandomForestClassifier()
model8 = AdaBoostClassifier()
model9 = GradientBoostingClassifier()

for clf in [model1, model2, model3,model4,model5]:
    train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
    train_sets.append(train_set)
    test_sets.append(test_set)
meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)

#使用决策树作为我们的次级分类器
dt_model.fit(meta_train, train_y)
model6.fit(meta_train,train_y)
model7.fit(meta_train,train_y)
model8.fit(meta_train,train_y)
model9.fit(meta_train,train_y)
df_predict = dt_model.predict(meta_test)
df_predict1=model6.predict(meta_test)
df_predict2=model6.predict(meta_test)
df_predict3=model6.predict(meta_test)
df_predict4=model6.predict(meta_test)


#投票决定最终结果，只要结果为0的出现了两次即以上我们即确定结果其为0，否则为1
last_result=[]
for i in range(0,len(df_predict)):
    lst=[]
    lst.append(df_predict[i])
    lst.append(df_predict1[i])
    lst.append(df_predict2[i])
    lst.append(df_predict3[i])
    lst.append(df_predict4[i])
    if(lst.count("0")>=2):
        last_result.append("0")
    else:
        last_result.append("1")
model_output("DecisionTreeClassifier",test_y,df_predict)
model_output("MLPClassifier",test_y,df_predict1)
model_output("RandomForestClassifier",test_y,df_predict2)
model_output("AdaBoostClassifier",test_y,df_predict3)
model_output("GradientBoostingClassifier",test_y,df_predict4)
model_output("复合投票结果",test_y,last_result)