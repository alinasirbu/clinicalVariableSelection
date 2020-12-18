#script that takes as command line parameters the fold number x (from 0 to 5 in our study) 
#and the number of features N to pass to the genetic algorithm
#the script reads training data for fold x from a file 'trainDataNoSCNoImagingFoldx.pkl'
#the script performs 10 runs, each with 10 fold internal cross validation, for each value of K in [50,75,100,125,150,175]

import csv
import pickle as pkl
import numpy as np
from sklearn.model_selection import cross_validate, cross_val_predict,LeaveOneOut, KFold
from sklearn import tree
from sklearn.externals.six import StringIO
#import graphviz
from sklearn.metrics import  roc_auc_score, classification_report,accuracy_score
from sklearn import linear_model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stat
from scipy.stats import ks_2samp, mode
import random as r
from datetime import datetime
from sklearn import preprocessing
import sys



#HELPER FUNCTIONS FOR THE ANALYSIS

def pred(accuracy,true):#extract prediction from loocv accuracy values, class 3 is deceased, calss 1 is dismissed
    if accuracy==1:
        return true
    else:
        if true==3:
            return 1
        return 3


def removeNanPatients(x,y):
    #select the patients that have 0 missing values
    patientIndices=np.sum(np.isnan(x),axis=1)<1
    xx=x[patientIndices,:]
    yy=y[patientIndices]
    return (xx,yy)


#FUNCTIONS FOR THE GENETIC ALGORITHM

#MUTATION
def mutate(pop,popSize,fitness,topmut,totalFeatures):
    for i in range(int(popSize*(1-topmut)),popSize):
        newIndividual=pop[i][:]
        mutatedGene=r.choice(range(len(pop[i])))
        newGene=r.choice(range(totalFeatures))
        newIndividual[mutatedGene]=newGene
        pop.append(newIndividual)
      
#CROSSOVER  
def cross(pop,popSize,fitness,ncross,topcross):     
    for i in range(ncross):
        child1=pop[r.choice(range(int(popSize*(1-topcross)),popSize))][:]
        child2=pop[r.choice(range(int(popSize*(1-topcross)),popSize))][:]
        cut=r.choice(range(len(child1)))
        for j in range(cut):
            g=child1[j]
            child1[j]=child2[j]
            child2[j]=g
        pop.append(child1)
        pop.append(child2)

#SELECTION
def select(pop,popSize,s):
    indices=[i for i in np.argsort([v[0] for v in s])[-popSize:]]
    return [pop[i] for i in indices],[s[i] for i in indices]


#HELPER FOR EVALUATION -  LEAVE ONE OUT CROSS VALIDATION WITH A SET OF CLINICAL VARIABLES
def loocv(fout,x,y,min_patients=100,verbose=True):
    xx,yy=removeNanPatients(x,y)
    if verbose:
        fout.write(str(xx.shape)+str(yy.shape)+"\n")
    if xx.shape[0]<min_patients or len(yy[yy==1])<3 or len(yy[yy==3])<3  : #too few patients or one class not present
        return (0,0,xx.shape[0])
    
    loo = LeaveOneOut()
    cv=loo.split(xx,yy)
    classifier=linear_model.LogisticRegression()
    results = cross_validate(classifier, xx, yy, cv=cv,return_estimator=True)
    y_pred=[pred(results['test_score'][i],yy[i]) for i in range(len(yy)) ] 
    AUC=roc_auc_score(yy,y_pred)
    ACC=accuracy_score(yy,y_pred)
    if verbose:
        fout.write ("AUC: "+str(AUC)+ " ACC: "+str(ACC)+"\n")
        fout.write (classification_report(yy,y_pred))
    return (AUC,ACC,xx.shape[0])

#EVALUATION - COMPUTE FITNESS AS AUC FROM LEAVE ONE OUT CROSS VALIDATION
def evaluateWithLoocv(fout,pop,x,y,currentFitness,min_patients):#current fitness contains fitness of first popsize elements
    start=len(currentFitness)#compute fitness only for remaining individuals
    for geneIndices in pop[start:]:
        result=loocv(fout,x[:,geneIndices],y,verbose=False,min_patients=min_patients)
        currentFitness.append((result[0],result[2]))
    return currentFitness


##### GENETIC ALGORITHM THAT RETURNS A SET OF popSize CLINICAL VARIABLES AND ITS AUC

def featureSelectionLoocv(fout,x,y,K,iterations, popSize,ncross,topmut,topcross,minPatients):
    #ga search of best clinical features to maximise AUC
    r.seed(datetime.now().microsecond)
    pop=[r.sample(range(x.shape[1]),k=K) for i in range(popSize)]
    #print "Evaluating initial population..."
    fitness=evaluateWithLoocv(fout,pop,x,y,[],minPatients)
    b=np.argmax([f[0] for f in fitness])
    bestF=fitness[b]
    bestIndices=pop[b][:]
    fout.write (str(bestF)+"\n")
    
    for i in range(iterations):
        #print "iteration ", i 
        mutate(pop,popSize,fitness,topmut,x.shape[1])
        cross(pop,popSize,fitness,ncross,topcross)
        fitness=evaluateWithLoocv(fout,pop,x,y,fitness,minPatients)
        b=np.argmax([f[0] for f in fitness])
        maxFc=fitness[b]
        geneIndices=pop[b][:]
        if maxFc[0]>bestF[0]:
            bestF=maxFc
            bestIndices=geneIndices[:]
            fout.write (str(bestF)+"\n")
        pop,fitness=select(pop,popSize,fitness)
    return bestIndices,bestF[0],bestF[1] 


#TEN FOLD CROSS VALIDATION - PERFORM FEATURE SELECTION ON TRAINING DATA, COMPUTE INTERNAL AUC ON TEST DATA
def classification(fout, x,y,featureNames,folds,nFeatures,iterations=10,minPatients=100,minAUC=0.75):
    ##k fold cross validation 
    popSize=100
    ncross=20
    topmut=0.3#mutation done only to a fraction of individuals at the top
    topcross=0.3#crossover done only among a fraction of individuals at the top
    cv = KFold(folds,True)
    cv=cv.split(x,y)
    y_pred=[]
    y_true=[]
    allFeatures={}
    for train_index, test_index in cv:
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #feature selection with GA on train data
        features, fsAUC,fsPatientCount=featureSelectionLoocv(fout,x_train,y_train,nFeatures,iterations,popSize,ncross,topmut,topcross,minPatients)
        fout.write("Feature selection AUC:"+str( fsAUC)+ " Patients: "+str(fsPatientCount)+"\n")
        fout.write ("Classifying with features "+str([featureNames[i] for i in features])+"\n")
        #use features for new classifier, train on train data, test on test data
        classifier=linear_model.LogisticRegression()
        xx_train,yy_train=removeNanPatients(x_train[:,features],y_train)
        if len(set(yy_train))>1:
            classifier.fit(xx_train,yy_train)
            xx_test,yy_test=removeNanPatients(x_test[:,features],y_test)
            if (len(yy_test)>0):
                y_pred_fold = classifier.predict(xx_test)  
                y_true=np.hstack((y_true,yy_test))
                y_pred=np.hstack((y_pred,y_pred_fold))
                AUC=-1
                if len(set(yy_test))==2:
                    AUC=roc_auc_score(yy_test,y_pred_fold)
                ACC=accuracy_score(yy_test,y_pred_fold)
                fout.write ("Internal FOLD AUC: "+str(AUC)+ " ACC: "+str(ACC)+ " Train size: "+str( xx_train.shape)+ " Test size: "+str( xx_test.shape)+"\n" )
                fout.flush()
                if AUC>minAUC or (AUC==-1 and ACC>minAUC):#consider the run only if reasonable on validation data
                    for f in features:
                        allFeatures[f]=allFeatures.get(f,0)+1
        else:
            fout.write ("Internal FOLD with one class only ")
    AUC=-1
    if len(set(y_true))==2:
        AUC=roc_auc_score(y_true,y_pred)
    ACC=accuracy_score(y_true,y_pred)
    fout.write( "Internal Total AUC: "+str(AUC)+ " ACC: "+str(ACC)+"\n" )
    fout.write ("Internal Feature summary: "+str( [(featureNames[i],allFeatures[i]) for i in allFeatures])+"\n")
    fout.write (classification_report(y_true,y_pred))
    return allFeatures



#SETUP ALGORITHM PARAMETERS FOR ANALYSIS

fold=sys.argv[1] #####PASS FOLD AS COMMAND LINE PARAMETER
nF=sys.argv[2] #####PASS NUMBER OF FEATURES AS COMMAND LINE PARAMETER


x,y,patients,features=pkl.load(open('trainDataNoSCNoImagingFold'+str(fold)+'.pkl','rb'))

x=preprocessing.scale(x)

print (x.shape)
print(y.shape)


minPatients=[50,75,100,125,150,175] #K - THE THRESHOLD ON THE NUMBER OF PATIENTS
runs=range(10)#REPEATED RUNS

#repeated runs
for run in runs:
    for mp in minPatients:
        fName="internalRunsOutputNF"+sys.argv[2]+"minPatients"+str(mp)+"fold"+str(fold)+"run"+str(run)
        print (fName)
        fout=open(fName+".txt",'w')
        allFeatures=classification(fout,x,y,features,folds=10,nFeatures=int(sys.argv[2]),iterations=20,minPatients=mp)#leave minauc default because not using now anyway
        fout.close()



