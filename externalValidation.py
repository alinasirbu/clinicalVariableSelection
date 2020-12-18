#script for eternal validation of features ranked by the feature selection algrotihm
#the script expects data already divided intro train and test, in different files
#builds a logistic regression model on the training data and tests on the test data


import csv
import numpy as np
from sklearn.model_selection import cross_validate, cross_val_predict,LeaveOneOut, KFold
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import  roc_auc_score, classification_report,accuracy_score
from sklearn import linear_model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stat
from scipy.stats import ks_2samp, mode
import random as r
import graphviz
from datetime import datetime
from sklearn import preprocessing
import pickle as pkl



def removeNanPatients(x,y):
    #select the patients that have 0 missing values
    patientIndices=np.sum(np.isnan(x),axis=1)<1
    xx=x[patientIndices,:]
    yy=y[patientIndices]
    return (xx,yy)



###INPUT FOLD NUMBER
fold=4
x_train,y_train,patients_train,features=pkl.load(open('trainDataFold'+str(fold)+'.pkl','rb'))


#load test data
x_test,y_test,patients_test,features=pkl.load(open('testDataFold'+str(fold)+'.pkl','rb'))


##read feature ranking from FS
logitfs=pkl.load(open('finalFeaturesFold'+str(fold)+'.pkl','rb'))


#train logistic regression models with increasing number of top features

train_aucs=[]
train_accs=[]
train_supps=[]
test_aucs=[]
test_accs=[]
test_supps=[]

models=20
for i in range(min(models,len(logitfs))):
    fts=[logitfs[j][0] for j in range(i+1) ]
    ftsIndices=[j for j in range(len(features)) if features[j] in fts]
    print (ftsIndices)
    x1_train=x_train[:,ftsIndices]
    x1_train,y1_train=removeNanPatients(x1_train,y_train)
    print (x1_train.shape)
    train_supps.append((len(y1_train[y1_train==1]),len(y1_train[y1_train==3])))
    print (fts)
    x1_test=x_test[:,ftsIndices]
    x1_test,y1_test=removeNanPatients(x1_test,y_test)
    print (x1_test.shape)
    test_supps.append((len(y1_test[y1_test==1]),len(y1_test[y1_test==3])))
    #cross validate classification
    classifier=linear_model.LogisticRegression(max_iter=2000)#,class_weight='balanced')
    classifier.fit(x1_train,y1_train)
    train_pred=classifier.predict(x1_train)
    test_pred=classifier.predict(x1_test)
    print(classification_report(y1_train,train_pred))
    print(classification_report(y1_test,test_pred))
    if len(set(y1_test))==2:
        test_aucs.append(roc_auc_score(y1_test,test_pred))
    else:
        test_aucs.append(-1)
    train_aucs.append(roc_auc_score(y1_train,train_pred))
    test_accs.append(accuracy_score(y1_test,test_pred))
    train_accs.append(accuracy_score(y1_train,train_pred))


print("Clinical variable & Train AUC & Train ~ Accuracy & Train ~ Support &Validation AUC & Validation ~~ Accuracy & Validation ~ Support \\\\ \\hline ")
for i in range(min(models,len(logitfs))):
    print (fts[i],
           "& %.2f"%train_aucs[i]," & %.2f &"%train_accs[i],train_supps[i][0]+train_supps[i][1]," (",train_supps[i][1],')',
           "& %.2f"%test_aucs[i]," & %.2f &"%test_accs[i], test_supps[i][0]+test_supps[i][1]," (",test_supps[i][1],')',"\\\\ \\hline",sep='')


