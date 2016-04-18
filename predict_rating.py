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
import matplotlib
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import csv
import matplotlib.pyplot as plt

def PredictRating():
	X=np.loadtxt('Count.txt',skiprows=1)
	Y=np.loadtxt('./InputData/Ratings.txt')
	X=X[0:6432]
	num=1000
	
	acc_lr=[]
	acc_lsvc=[]
	for i in range(1,7):
		X1=X[0:num]
		Y1=Y[0:num]
		[pred_acc_lr,pred_acc_lsvc]=callClassifiersForPrediction(X1,Y1)
		num+=1000
		acc_lr.append(pred_acc_lr)
		acc_lsvc.append(pred_acc_lsvc)
	plt.figure(1)
	y_points=[1000,2000,3000,4000,5000,6000]
	plt.plot(y_points,acc_lr,linestyle='--',marker='o',markersize=8,fillstyle='right',alpha=0.6, label='Logistic Regression',color='red')
	plt.hold(True)
	plt.plot(y_points,acc_lsvc,linestyle='--',marker='^',markersize=8,fillstyle='right',alpha=0.6, label='Linear SVC',color='blue')
	plt.xlabel("Dataset size")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs Dataset size")
	plt.legend(loc='best')
	plt.show()
	X1=X[0:2000]
	Y1=Y[0:2000]
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
	plt.figure(2)
	plt.plot(C_list,lsvc_acc,linestyle='--',marker='o',markersize=8,fillstyle='right',alpha=0.6, label='LinearSVC Accuracy',color='red')
	plt.hold(True)
	plt.xscale('log')
	plt.xlabel("log of Regularization parameter")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs Regularization parameter")
	plt.legend(loc='best')
	plt.show()
	plt.figure(3)
	plt.plot(C_list,lsvc_mse,linestyle='--',marker='o',markersize=8,fillstyle='right',alpha=0.6, label='LinearSVC MSE',color='red')
	plt.hold(True)
	plt.xscale('log')
	plt.xlabel("log of Regularization parameter")
	plt.ylabel("MSE")
	plt.title("MSE vs Regularization parameter")
	plt.legend(loc='best')
	plt.show()
	
	
def callClassifiersForPrediction(X,Y):
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
		model = LogisticRegression(multi_class='multinomial', solver='lbfgs',C=0.001)
		model.fit(X_train, y_train)
		predicted= model.predict(X_test)
		print "Test LR accuracy-",accuracy_score(y_test, predicted), " of test set number 1"
		print "Test LR mse-",mean_squared_error(y_test, predicted), " of test set number 1"
		model1 = LinearSVC(C=0.01)
		model1.fit(X_train, y_train)
		predicted1= model1.predict(X_test)
		print "Test Linear SVC accuracy-",accuracy_score(y_test, predicted1)
		print "Test Linear SVC mse-",mean_squared_error(y_test, predicted1)
		return accuracy_score(y_test, predicted),accuracy_score(y_test, predicted1)