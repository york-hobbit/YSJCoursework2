# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 08:56:22 2023

@author: PWade
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix



#load data
#read data
data = pd.read_csv("dementia.csv")
data.head()

#Data exploration and cleaning
#find information about the data
data.info()
unique = data.nunique()
unique

#check for nan
check_nan = data.isnull()
check_nan
#SES MMSE have nan values

sns.set_palette("bright")

sns.countplot(x = data["Group"])

sns.countplot(x = data["Visit"], hue = data["Group"])

sns.countplot(x = data["Group"], hue = data["M/F"])

sns.pairplot(data=data, hue = "Group")

sns.displot(data=data, x="Age", hue="Group",
            multiple="stack", kind="kde")

sns.displot(data=data, x="Age", hue="M/F",
            multiple="stack", kind="kde")

sns.displot(data=data, x="Age", hue="Group", col="Group")


#drop Subject_ID, MRI_ID and Hand
data = data.drop(["Subject ID", "MRI ID", "Hand"], axis=1)

#change M/F and Group to numerical values
data["M/F"] = data["M/F"].replace(["M", "F"], ["0", "1"])
data["Group"] = data["Group"].replace(["Nondemented", "Demented", "Converted"], ["0", "1", "2"])

#check data types
data.info()


#convert Group and M/F to int
data["M/F"] = data["M/F"].astype(int)
data["Group"] = data["Group"].astype(int)

#Find data ranges
desc = data.describe()
desc

#replace nan MMSE with mode (catogorical) replace ses mean 

#data["MMSE"] = data["MMSE"].fillna(data["MMSE"].mode())
#drope mode did work so will drop rows for MMSE NaN
data=data.dropna(subset=["MMSE"])
data["SES"] = data["SES"].fillna(data["SES"].mean())


data.isnull().any()
#convert Group and M/F to int
data["M/F"] = data["M/F"].astype(int)
data["Group"] = data["Group"].astype(int)
data.info()


#split data and lables
X = data.drop(["Group"], axis=1)
y = data["Group"]


#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



columnn_names = X_train.columns

feature_imp = pd.Series(clf.feature_importances_, index=columnn_names).sort_values(ascending=False)

#bar plot of features of importance

sns.barplot(x=feature_imp, y=feature_imp.index)


plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Important Features")


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d').set_title('Demetia Prediction confusion matrix (0 = Nondemented, 1 = Demented, 2 = Converted)')

print(classification_report(y_test,y_pred))

#Drop EDUC, SES, Visit, M/F

X = data.drop(["EDUC", "SES", "Visit", "M/F"], axis=1)

#run again with new X values
#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d').set_title('Demetia Prediction confusion matrix 2 (0 = Nondemented, 1 = Demented, 2 = Converted)')
print(classification_report(y_test,y_pred))

#make adjustments to default classifier
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=75, max_depth=3, min_samples_leaf=3)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d').set_title('Demetia Prediction confusion matrix 3 (0 = Nondemented, 1 = Demented, 2 = Converted)')
print(classification_report(y_test,y_pred))

#make adjustments to default classifier
#Create a Gaussian Classifier
clf=RandomForestClassifier(criterion="entropy", n_estimators=75, max_depth=3, min_samples_leaf=3)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d').set_title('Demetia Prediction confusion matrix 4 (0 = Nondemented, 1 = Demented, 2 = Converted)')
print(classification_report(y_test,y_pred))

#make adjustments to default classifier
#Create a Gaussian Classifier
clf=RandomForestClassifier(bootstrap=False, n_estimators=75, max_depth=3, min_samples_leaf=3)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d').set_title('Demetia Prediction confusion matrix 5 (0 = Nondemented, 1 = Demented, 2 = Converted)')
print(classification_report(y_test,y_pred))


