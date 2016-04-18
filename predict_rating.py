'''
Created on April 17, 2016

@author: inderjot, aprajita
'''
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold as KFold
import matplotlib
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import csv
import matplotlib.pyplot as plt

#function for rating prediction
def PredictRating():
	X=np.loadtxt('Count.txt',skiprows=1)
	Y=np.loadtxt('./InputData/Ratings.txt')
	X=X[0:6432]
	num=1000
	acc_lr=[]
	acc_lsvc=[]
	acc_ssvc=[]
	
	#for loop to choose best dataset size
	for i in range(1,7):
		X1=X[0:num]
		Y1=Y[0:num]
		[pred_acc_lr,pred_acc_lsvc,pred_acc_ssvc]=callClassifiersForPrediction(X1,Y1)
		num+=1000
		acc_lr.append(pred_acc_lr)
		acc_lsvc.append(pred_acc_lsvc)
		acc_ssvc.append(pred_acc_ssvc)
	
	#plot dataset size vs accuracy
	plt.figure(1)
	y_points=[1000,2000,3000,4000,5000,6000]
	plt.plot(y_points,acc_lr,linestyle='--',marker='o',markersize=8,fillstyle='right',alpha=0.6, label='Logistic Regression',color='red')
	plt.hold(True)
	plt.plot(y_points,acc_lsvc,linestyle='--',marker='^',markersize=8,fillstyle='right',alpha=0.6, label='Linear SVC',color='blue')
	plt.hold(True)
	plt.plot(y_points,acc_ssvc,linestyle='--',marker='*',markersize=8,fillstyle='right',alpha=0.6, label='Sigmoid SVC',color='green')
	plt.xlabel("Dataset size")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs Dataset size")
	plt.legend(loc='best')
	plt.show()
	X1=X[0:2000]
	Y1=Y[0:2000]
	
	#Divide into training and testing set
	X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.20, random_state=42)
	C_list = [0.01, 0.1, 1, 10, 100]
	lsvc_acc=[]
	lsvc_mse=[]
	for num in C_list:
		model = LinearSVC(C=num)
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)
		print "Test Linear SVC accuracy-",accuracy_score(y_test, predicted), " with C=",num
		print "Test Linear SVC mse-",mean_squared_error(y_test, predicted), " with C=",num
		lsvc_acc.append(accuracy_score(y_test, predicted))
		lsvc_mse.append(mean_squared_error(y_test, predicted))
		
	#plot accuracy vs regularization parameter
	plt.figure(2)
	plt.plot(C_list,lsvc_acc,linestyle='--',marker='o',markersize=8,fillstyle='right',alpha=0.6, label='Linear SVC Accuracy',color='red')
	plt.hold(True)
	plt.xscale('log')
	plt.xlabel("log of Regularization parameter")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs Regularization parameter")
	plt.legend(loc='best')
	plt.show()
	
	#plot mse vs regularization parameter
	plt.figure(3)
	plt.plot(C_list,lsvc_mse,linestyle='--',marker='o',markersize=8,fillstyle='right',alpha=0.6, label='Sigmoid SVC MSE',color='red')
	plt.hold(True)
	plt.xscale('log')
	plt.xlabel("log of Regularization parameter")
	plt.ylabel("MSE")
	plt.title("MSE vs Regularization parameter")
	plt.legend(loc='best')
	plt.show()
	folds = KFold(len(Y1), n_folds=5)
	count = 0
	
	#5-fold cross validation
	for train_index,test_index in folds:
		count+=1
		X_train,y_train=X1[train_index],Y1[train_index]
		X_test,y_test=X1[test_index],Y1[test_index]
		model = SVC(kernel='sigmoid',C=0.01)
		model.fit(X_train, y_train)
		predicted= model.predict(X_test)
		print "Test Sigmoid SVC accuracy-",accuracy_score(y_test, predicted), " of test set number:",count
		print "Test Sigmoid SVC mse-",mean_squared_error(y_test, predicted), " of test set number:",count

	
def callClassifiersForPrediction(X,Y):
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
		
		#Multinomial Logistic Regression
		model = LogisticRegression(multi_class='multinomial', solver='lbfgs',C=0.001)
		model.fit(X_train, y_train)
		predicted= model.predict(X_test)
		print "Test LR accuracy-",accuracy_score(y_test, predicted), " of test set number 1"
		print "Test LR mse-",mean_squared_error(y_test, predicted), " of test set number 1"
		
		#Linear SVM
		model1 = LinearSVC(C=0.01)
		model1.fit(X_train, y_train)
		predicted1= model1.predict(X_test)
		print "Test Linear SVC accuracy-",accuracy_score(y_test, predicted1)
		print "Test Linear SVC mse-",mean_squared_error(y_test, predicted1)
		
		#Sigmoid SVM
		model2 = SVC(kernel='sigmoid',C=0.01)
		model2.fit(X_train, y_train)
		predicted2= model2.predict(X_test)
		print "Test Sigmoid kernel SVC accuracy-",accuracy_score(y_test, predicted2)
		print "Test Sigmoid kernel SVC mse-",mean_squared_error(y_test, predicted2)
		return accuracy_score(y_test, predicted),accuracy_score(y_test, predicted1),accuracy_score(y_test, predicted2)