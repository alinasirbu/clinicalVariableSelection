#this script takes the output of the featureSelection script


import matplotlib
import pylab as pl
import numpy as np

import csv
import pickle


nFeatures=[5,10]
minPatients=[50,75,100,125,150,175]
runs=[0,1,2,3,4,5,6,7,8,9]

fold=0#change this for your fold
minAUC=0.85
minACC=0.85



##visualise internal validation AUC over 10 runs

internalAUCs=[] # holds one array for each possible parameter combination (nf,mp)
internalACCs=[] # holds one array for each possible parameter combination (nf,mp)

labels=[(nf,mp) for nf in nFeatures for mp in minPatients]
for nf in nFeatures:
    for mp in minPatients:
        aucs=[]
        accs=[]
        for r in runs:
                f=open("internalRunsOutputNF"+str(nf)+"minPatients"+str(mp)+"fold"+str(fold)+"run"+str(r)+".txt")
                for line in f:
                    if line.startswith("Internal FOLD AUC"):
                        tokens=line.split(' ')
                        auc=float(tokens[3])
                        acc=float(tokens[5])
                        if auc!=-1:
                            aucs.append(auc)
                        accs.append(acc)
        internalAUCs.append(aucs)
        internalACCs.append(accs)

pl.figure(figsize=(5,2.5))
pl.boxplot(internalAUCs,labels=labels)
pl.xticks(rotation=90)
pl.xlabel("GA parameters: (N, p*)")
pl.ylabel("Internal AUC")
pl.grid()
pl.subplots_adjust(bottom=0.3)
pl.savefig("internalAUC.pdf")
pl.show()



##combine internal features, by selecting only the sets with internal validation AUC>threshold above

allfs={}
count=0
for nf in nFeatures:
    for mp in minPatients:
            for r in runs:
                f=open("internalRunsOutputNF"+str(nf)+"minPatients"+str(mp)+"fold"+str(fold)+"run"+str(r)+".txt")
                prevLine=''
                for line in f:
                    if line.startswith("Internal FOLD AUC"):
                        tokens=line.split(' ')
                        auc=float(tokens[3])
                        acc=float(tokens[5])
                        if auc>minAUC or (auc==-1 and acc>minACC):##add to features
                            count=count+1
                            strFeatures=prevLine.split('[')[1].strip('\n]').split(",")
                            strFeatures=[ft.strip("' ") for ft in strFeatures]
                            strFeatures=list(set(strFeatures))
                            #print(strFeatures)
                            for ft in strFeatures:
                                allfs[ft]=allfs.get(ft,0)+1
                    prevLine=line
                                
allfs=[(k,allfs[k]/count) for k in allfs]

allfs.sort(key=lambda x:-x[1])
allfs


#dump the ranking of features in a file - this is the final result of our algorithm
with open('finalFeaturesFold'+str(fold)+'.pkl','wb') as f :
    pickle.dump(allfs,f)



