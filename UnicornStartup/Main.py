from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

main = tkinter.Tk()
main.title("Startup Unicorn Prediction Using Advanced Machine Learning Algorithms") #designing main screen
main.geometry("1300x1200")

global filename
global X, Y
global X_train, Y_train
global classifier
global accuracy, precision, recall, fscore
global X_train, X_test, y_train, y_test
global dataset

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")#uploading dataset
    dataset = pd.read_csv(filename) #reading dataset from loaded file
    dataset.fillna(0, inplace = True)
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    text.insert(END,str(dataset.head()))
    label = dataset.groupby('labels').size()
    label.plot(kind="bar")
    plt.show()

def preprocessDataset():
    global dataset
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)

    X = dataset[['age_first_funding_year', 'age_last_funding_year', 'relationships', 'funding_rounds', 
           'funding_total_usd', 'milestones', 'is_CA', 'is_NY', 'is_MA', 'is_TX', 
           'is_otherstate', 'is_software', 'is_web', 'is_mobile', 'is_enterprise', 
           'is_advertising', 'is_gamesvideo', 'is_ecommerce', 'is_biotech', 
           'is_consulting', 'is_othercategory', 'has_VC', 'has_angel', 'has_roundA',
           'has_roundB', 'has_roundC', 'has_roundD', 'avg_participants', 
           'is_top500', 'age_first_milestone_year','age_last_milestone_year','labels'
           ]]
    Y = dataset['labels']
    X = X.values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END,"Total records available in dataset: "+str(X.shape[0])+"\n")
    text.insert(END,"Total features/columns found in dataset: "+str(X.shape[1])+"\n\n")
    text.insert(END,"Dataset train and test split details\n\n")
    text.insert(END,"Training dataset split 80% total records are: "+str(X_train.shape[0])+"\n")
    text.insert(END,"Testing dataset split 20% total records are: "+str(X_test.shape[0])+"\n")


def calculateMetrics(y_test,predict,name):
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)    
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,name+" Accuracy  : "+str(a)+"\n")
    text.insert(END,name+" Precision : "+str(p)+"\n")
    text.insert(END,name+" Recall    : "+str(r)+"\n")
    text.insert(END,name+" FSCORE    : "+str(f)+"\n\n")

def runDecisionTree():
    global accuracy, precision, recall, fscore
    global X_train, X_test, y_train, y_test
    accuracy = []
    precision = []
    recall = []
    fscore = []
    text.delete('1.0', END)
    dt_cls = DecisionTreeClassifier(max_depth=2,criterion="entropy") 
    dt_cls.fit(X_train, y_train)
    predict = dt_cls.predict(X_test)
    calculateMetrics(y_test,predict,"Decision Tree Algorithm")

def runRF():
    global classifier
    global accuracy, precision, recall, fscore
    global X_train, X_test, y_train, y_test
    rf_cls = RandomForestClassifier(n_estimators=2,criterion="entropy",max_features="sqrt") 
    rf_cls.fit(X_train, y_train)
    predict = rf_cls.predict(X_test)
    calculateMetrics(y_test,predict,"Random Forest Algorithm")
    classifier = rf_cls

    svm_cls = svm.SVC(probability=True)
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics(y_test,predict,"SVM Algorithm")

def runGB():
    global accuracy, precision, recall, fscore
    global X_train, X_test, y_train, y_test
    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)
    predict = gb.predict(X_test)
    calculateMetrics(y_test,predict,"Gradient Boosting Algorithm")

def graph():
    df = pd.DataFrame([['Decision Tree','Precision',precision[0]],['Decision Tree','Recall',recall[0]],['Decision Tree','F1 Score',fscore[0]],['Decision Tree','Accuracy',accuracy[0]],
                       ['Random Forest','Precision',precision[1]],['Random Forest','Recall',recall[1]],['Random Forest','F1 Score',fscore[1]],['Random Forest','Accuracy',accuracy[1]],
                       ['SVM','Precision',precision[2]],['SVM','Recall',recall[2]],['SVM','F1 Score',fscore[2]],['SVM','Accuracy',accuracy[2]],
                       ['Gradient Boosting','Precision',precision[3]],['Gradient Boosting','Recall',recall[3]],['Gradient Boosting','F1 Score',fscore[3]],['Gradient Boosting','Accuracy',accuracy[3]],
 

                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def predict():
    text.delete('1.0', END)
    global classifier
    testfile = filedialog.askopenfilename(initialdir = "Dataset")
    testdata = pd.read_csv(testfile)
    testdata.fillna(0, inplace = True)
    testdata = testdata[['age_first_funding_year', 'age_last_funding_year', 'relationships', 'funding_rounds', 
           'funding_total_usd', 'milestones', 'is_CA', 'is_NY', 'is_MA', 'is_TX', 
           'is_otherstate', 'is_software', 'is_web', 'is_mobile', 'is_enterprise', 
           'is_advertising', 'is_gamesvideo', 'is_ecommerce', 'is_biotech', 
           'is_consulting', 'is_othercategory', 'has_VC', 'has_angel', 'has_roundA',
           'has_roundB', 'has_roundC', 'has_roundD', 'avg_participants', 
           'is_top500', 'age_first_milestone_year','age_last_milestone_year','labels'
           ]]
    testdata = testdata.values
    predict = classifier.predict(testdata)
    for i in range(len(testdata)):
        if predict[i] == 1:
            text.insert(END,"Startup Test Values: "+str(testdata[i])+" =====> Predicted As SUCCESS\n\n")
        else:
            text.insert(END,"Startup Test Values: "+str(testdata[i])+" =====> Predicted As FAILURE\n\n")
        

font = ('times', 16, 'bold')
title = Label(main, text="Startup Unicorn Prediction Using Advanced Machine Learning Algorithms")
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Startup Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=360,y=550)
preprocessButton.config(font=font1) 

dtButton = Button(main, text="Run Decision Tree Algorithm", command=runDecisionTree)
dtButton.place(x=580,y=550)
dtButton.config(font=font1) 

rfButton = Button(main, text="Run SVM & Random Forest Algorithm", command=runRF)
rfButton.place(x=50,y=600)
rfButton.config(font=font1)

gbButton = Button(main, text="Run Gradient Boosting Algorithm", command=runGB)
gbButton.place(x=360,y=600)
gbButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=650)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Startup Status from Test Data", command=predict)
predictButton.place(x=360,y=650)
predictButton.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
