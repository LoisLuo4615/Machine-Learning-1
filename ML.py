#!usr/bin/env python    
#-*- coding: utf-8 -*-    
    
import sys    
import os    
import time    
from sklearn import metrics    
import numpy as np    
import pickle  
import importlib  
    
importlib.reload(sys)    
#sys.setdefaultencoding('utf8')  
  
# Multinomial Naive Bayes Classifier  
# 朴素贝叶斯  
def naive_bayes_classifier(train_x, train_y):    
    from sklearn.naive_bayes import MultinomialNB    
    model = MultinomialNB(alpha=0.01)    
    model.fit(train_x, train_y)    
    return model    
    
    
# KNN Classifier  
# K最近邻  
def knn_classifier(train_x, train_y):    
    from sklearn.neighbors import KNeighborsClassifier    
    model = KNeighborsClassifier()    
    model.fit(train_x, train_y)    
    return model    
    
    
# Logistic Regression Classifier  
# 逻辑回归  
def logistic_regression_classifier(train_x, train_y):    
    from sklearn.linear_model import LogisticRegression    
    model = LogisticRegression(penalty='l2')    
    model.fit(train_x, train_y)    
    return model    
    
    
# Random Forest Classifier  
# 随机森林  
def random_forest_classifier(train_x, train_y):    
    from sklearn.ensemble import RandomForestClassifier    
    model = RandomForestClassifier(n_estimators=8)    
    model.fit(train_x, train_y)    
    return model    
    
    
# Decision Tree Classifier  
# 决策树  
def decision_tree_classifier(train_x, train_y):    
    from sklearn import tree    
    model = tree.DecisionTreeClassifier()    
    model.fit(train_x, train_y)    
    return model    
    
    
# GBDT(Gradient Boosting Decision Tree) Classifier  
# 梯度推进   
def gradient_boosting_classifier(train_x, train_y):    
    from sklearn.ensemble import GradientBoostingClassifier    
    model = GradientBoostingClassifier(n_estimators=200)    
    model.fit(train_x, train_y)    
    return model    
    
    
# SVM Classifier  
# 支持向量机  
def svm_classifier(train_x, train_y):    
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    model.fit(train_x, train_y)    
    return model    
    
# SVM Classifier using cross validation  
# 支持向量机 交叉验证  
def svm_cross_validation(train_x, train_y):    
    from sklearn.grid_search import GridSearchCV    
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in best_parameters.items():    
        print (para, val  )  
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(train_x, train_y)    
    return model    
    
def read_data(data_file):    
    import gzip    
    f = gzip.open(data_file, "rb",'utf8')  
      
      
    train, val, test = pickle.load(f,encoding="bytes")  # add ,encoding="bytes"  
    f.close()    
    train_x = train[0]    
    train_y = train[1]    
    test_x = test[0]    
    test_y = test[1]    
    return train_x, train_y, test_x, test_y    
        
if __name__ == '__main__':    
    data_file = "C:\Python34\TestCodes\mnist.pkl.gz"    
    thresh = 0.5    
    model_save_file = None    
    model_save = {}    
        
    test_classifiers = ['NB 朴素贝叶斯', 'KNN K最近邻', 'LR  逻辑回归', 'RF  随机森林', 'DT 决策树', 'SVM 支持向量机', 'GBDT 梯度推进']    
    classifiers = {'NB':naive_bayes_classifier,        # 朴素贝叶斯  
                  'KNN':knn_classifier,                # K最近邻  
                   'LR':logistic_regression_classifier,# 逻辑回归   
                   'RF':random_forest_classifier,      # 随机森林  
                   'DT':decision_tree_classifier,      # 决策树  
                  'SVM':svm_classifier,                # 支持向量机  
                'SVMCV':svm_cross_validation,          # 支持向量机 交叉验证  
                 'GBDT':gradient_boosting_classifier   # 梯度推进   
    }    
        
    print("reading training and testing data...")  
    train_x, train_y, test_x, test_y = read_data(data_file)    
    num_train, num_feat = train_x.shape    
    num_test, num_feat = test_x.shape    
    is_binary_class = (len(np.unique(train_y)) == 2)    
    print ('******************** Data Info *********************' )   
    print ('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat)  )  
        
    for classifier in test_classifiers:    
        print ('******************* %s ********************' % classifier  )  
        start_time = time.time()    
        model = classifiers[classifier](train_x, train_y)    
        print ('training took %fs!' % (time.time() - start_time)  )  
        predict = model.predict(test_x)    
        if model_save_file != None:    
            model_save[classifier] = model    
        if is_binary_class:    
            precision = metrics.precision_score(test_y, predict)    
            recall = metrics.recall_score(test_y, predict)    
            print ('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall) )   
        accuracy = metrics.accuracy_score(test_y, predict)    
        print ('accuracy: %.2f%%' % (100 * accuracy)   )  
    
    if model_save_file != None:    
        pickle.dump(model_save, open(model_save_file, 'wb'))
