#!/usr/bin/env python3
import csv
import math
import numpy as np

runs = 0
total_runs = 10
accuracy = []
while(runs < total_runs):
	raw_data = np.genfromtxt("pima-indians-diabetes.csv", delimiter = ",", usecols = (1, 2, 3, 8))
	np.random.shuffle(raw_data) #randomize raw data

	train_data = raw_data[0:len(raw_data)//2] #separate 1st half to training data
	test_data = raw_data[len(raw_data)//2:] #separate 2nd half to test data
	correct = 0
	wrong = 0

	train_data_filter1 = train_data[:, 3] = train_data[:, 3] == 1 #setup filter where training data column 4 = 1
	mean1 = train_data[train_data_filter1].mean(0) #calculate mean for each attribute in training data
	mean1 = np.delete(mean1, 3, 0) #delete the 4th column from mean vector
	var1 = np.cov(train_data[train_data_filter1].transpose()) #calculate covariance matrix on training data
	var1 = np.delete(var1, 3, 1) #delete column 4 from covariance matrix
	var1 = np.delete(var1, 3, 0) #delete row 4 from covariance matrix

	train_data_filter0 = train_data[:, 3] = train_data[:, 3] == 0 #setup filter where training data column 4 = 0
	mean0 = train_data[train_data_filter0].mean(0)
	mean0 = np.delete(mean0, 3, 0)
	var0 = np.cov(train_data[train_data_filter0].transpose())
	var0 = np.delete(var0, 3, 1)
	var0 = np.delete(var0, 3, 0)

	priortmp1 = train_data[train_data_filter1].shape[0] #number of training data records where column 4 = 1
	priortmp0 = train_data[train_data_filter0].shape[0] #number of training data records where column 4 = 0

	#calculate priors
	prior0 = float(priortmp0) / (priortmp0 + priortmp1)
	prior1 = float(priortmp1) / (priortmp0 + priortmp1)

	for i in test_data:
		#calculate the independent likelihoods
		lklhood00 = np.exp(-(i[0] - mean0[0]) * (i[0] - mean0[0]) / (2 * var0[0,0])) / math.sqrt(var0[0,0])
		lklhood01 = np.exp(-(i[1] - mean0[1]) * (i[1] - mean0[1]) / (2 * var0[1,1])) / math.sqrt(var0[1,1])
		lklhood02 = np.exp(-(i[2] - mean0[2]) * (i[2] - mean0[2]) / (2 * var0[2,2])) / math.sqrt(var0[2,2])

		lklhood10 = np.exp(-(i[0] - mean1[0]) * (i[0] - mean1[0]) / (2 * var1[0,0])) / math.sqrt(var1[0,0])
		lklhood11 = np.exp(-(i[1] - mean1[1]) * (i[1] - mean1[1]) / (2 * var1[1,1])) / math.sqrt(var1[1,1])
		lklhood12 = np.exp(-(i[2] - mean1[2]) * (i[2] - mean1[2]) / (2 * var1[2,2])) / math.sqrt(var1[2,2])

		#calculate the posteriors
		post0 = prior0 * lklhood00 * lklhood01 * lklhood02
		post1 = prior1 * lklhood10 * lklhood11 * lklhood12

		#make decisions based on the posterior and count correct vs wrong on test data
		if(post0 > post1 and i[3] == 0):
			correct += 1
		elif(post1 < post0 and i[3] == 1):
			correct += 1
		else:
			wrong += 1

	runs += 1
	accuracy.append(100 * float(correct) / (correct + wrong))

print("Accuracy Average: " + str(np.average(accuracy)))
print("Accuracy Standard Deviation: " + str(np.std(accuracy)))
