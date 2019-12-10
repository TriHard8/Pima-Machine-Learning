#!/usr/bin/env python3

import csv
import math
import numpy as np

runs = 0
total_runs = 10
accuracy = []

while (runs < total_runs):
	raw_data = np.genfromtxt("pima-indians-diabetes.csv", delimiter = ",", usecols = (1, 2, 3, 8))
	np.random.shuffle(raw_data) #randomize dataset

	train_data = raw_data[0:len(raw_data)//2] #assign 1st half of data to training data
	test_data = raw_data[len(raw_data)//2:] #assign 2nd half of data to test data
	correct = 0
	wrong = 0

	train_data_filter1 = train_data[:, 3] = train_data[:, 3] == 1 #setting filter for training data when column 9 = 1
	mean1 = train_data[train_data_filter1].mean(0) #calculate mean of each attribute
	mean1 = np.delete(mean1, 3, 0) #remove the 4th (classification) column from mean vector
	var1 = np.cov(train_data[train_data_filter1].transpose()) #calculate covariance matrix of training data
	var1 = np.delete(var1, 3, 1) #removes the 4th column from the covariance matrix
	var1 = np.delete(var1, 3, 0) #removes the 4th row from the covariance matrix

	train_data_filter0 = train_data[:, 3] = train_data[:, 3] == 0
	mean0 = train_data[train_data_filter0].mean(0)
	mean0 = np.delete(mean0, 3, 0) #remove the 4th (classification) column from mean vector
	var0 = np.cov(train_data[train_data_filter0].transpose())
	var0 = np.delete(var0, 3, 1) #remove the 4th column from the covariance matrix
	var0 = np.delete(var0, 3, 0) #remove the 4th row from teh covariance matrix

	priortmp1 = train_data[train_data_filter1].shape[0] #gives the count where training data column 4 equals 1
	priortmp0 = train_data[train_data_filter0].shape[0] #gives the count where training data column 4 equals 0

	prior0 = float(priortmp0) / (priortmp0 + priortmp1) #calculates prior probability of training data where column 4 equals 0
	prior1 = float(priortmp1) / (priortmp0 + priortmp1) #calculates prior probability of training data where column 4 equals 1

	constant1 = 1 / math.sqrt(np.linalg.det(var1)) #calculates 1/sqrt(covariance)
	constant0 = 1 / math.sqrt(np.linalg.det(var0))
	inv_var0 = np.linalg.inv(var0) #inverse of covariance matrix
	inv_var1 = np.linalg.inv(var1)

	for i in test_data:
		j = np.delete(i, 3, 0)
		diff0 = j - mean0 #subtracts the training mean (classification == 0) from current record
		lklhood0 = constant0 * np.exp(-np.dot(np.dot(diff0, inv_var0), diff0) / 2) #calculates likelihood of classifcation == 0

		diff1 = j - mean1
		lklhood1 = constant1 * np.exp(-np.dot(np.dot(diff1, inv_var1), diff1) / 2) #calculates likelihood of classification == 1

		post0 = prior0 * lklhood0 #calculate posterior of classification == 0
		post1 = prior1 * lklhood1

		#Classify as whichever posterior is greater. This conditional determines if classification is correct.
		if(post0 > post1 and i[3] == 0):
			correct += 1
		elif(post1 < post0 and i[3] == 1):
			correct += 1
		else:
			wrong += 1

	accuracy.append(100 * float(correct) / (correct + wrong))
	runs += 1

print("Average Accuracy: " + str(np.average(accuracy)))
print("Standard Deviation of Accuracy: " + str(np.std(accuracy)))



