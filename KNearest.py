import csv
import math
import numpy as np
from scipy import spatial

runs = 0
accuracy = []
while(runs < 10):
	raw_data = np.genfromtxt("pima-indians-diabetes.csv", delimiter = ",", usecols = (1, 2, 3, 8))
	np.random.shuffle(raw_data) #shuffle data
	num_samples = 111 #value of k

	train_data = raw_data[0:len(raw_data)/2] #separate 1st half of raw data into training data
	test_data = raw_data[len(raw_data)/2:] #separate 2nd half of raw data into test data
	correct = 0
	wrong = 0

	train = np.delete(train_data, 3, 1) #assign train to train_data without column 4
	tree = spatial.KDTree(train) #create kd tree with training data

	for i in test_data:
		distance, closest = tree.query(np.delete(i, 3, 0), k = num_samples) #find k nearest neighbor(s) and assign it/them to closest
		post0 = 0
		post1 = 0
		j = 0
		#closest will be a list if k is greater than 1 and integer if equal to 1, so this conditional separates them
		if(num_samples > 1):
			while(j < num_samples): #this checks each of the closest neighbor
				if(train_data[closest[j], 3] == 1):
					post1 += 1
				else:
					post0 += 1
				j += 1		
		else:
			if(train_data[closest, 3] == 1):
				post1 += 1
			else:
				post0 += 1

		if(post1 > post0 and i[3] == 1):
			correct += 1
		elif(post0 > post1 and i[3] == 0):
			correct += 1
		else:
			wrong += 1

	runs += 1
	accuracy.append(100 * float(correct) / (correct + wrong))

print "Accuracy Average: " + str(np.average(accuracy))
print "Accuracy Standard Deviation: " + str(np.std(accuracy))
