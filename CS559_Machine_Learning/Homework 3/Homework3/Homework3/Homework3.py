import csv
import math
import numpy as np
import numpy.matlib

runs = 0
accuracy = []
while (runs < 10):
    correct = 0
    wrong = 0
    raw_data = np.genfromtxt("pima-indians-diabetes.csv", delimiter = ",", usecols = (0, 1, 2, 3, 4, 5, 6, 7, 8))
    np.random.shuffle(raw_data) #randomize data
    train_data = raw_data[0:len(raw_data)/2] #assign 1st half of data to training data
    test_data = raw_data[len(raw_data)/2:] #assign 2nd half of data to test data
    col_8 = train_data[:, 8]
    fixed = np.zeros((len(col_8))) #for some reason col_8 was being changed randomly on line 35, so this was created to fix this issue, since I didn't have time to resolve the why.
    for i in range(0, len(fixed)):
        fixed[i] = col_8 [i]
    col_8_test = test_data[:, 8]

    mean = train_data.mean(0) #overall mean
    mean = np.delete(mean, 8, 0)

    train_data_filter1 = train_data[:, 8] = train_data[:, 8] == 1 #setting filter for training data when column 9 = 1
    mean1 = train_data[train_data_filter1].mean(0) #calculate mean of each attribute
    mean1 = np.delete(mean1, 8, 0) #remove the 9th column from mean vector
    N1 = len(train_data[train_data_filter1])
    S1 = np.cov(train_data[train_data_filter1].transpose()) #scatter matrix for class 1
    S1 = np.delete(S1, 8, 0)
    S1 = np.delete(S1, 8, 1)

    train_data_filter0 = train_data[:, 8] = train_data[:, 8] == 0
    mean0 = train_data[train_data_filter0].mean(0)
    mean0 = np.delete(mean0, 8, 0)
    N0 = len(train_data[train_data_filter0])
    S0 = np.cov(train_data[train_data_filter0].transpose()) #scatter matrix for class 0
    S0 = np.delete(S0, 8, 0)
    S0 = np.delete(S0, 8, 1)
    
    Sw = S1 + S0 #Within class scatter matrix
    Sw_inv = np.linalg.inv(Sw)

    mean = mean.reshape(8,1)
    mean0 = mean0.reshape(8,1)
    mean1 = mean0.reshape(8,1)
    Sb = N1 * np.dot((mean1 - mean), (mean1 - mean).transpose()) + N0 * np.dot((mean0 - mean), (mean0 - mean).transpose()) #between class scatter matrix

    Sw_inv_Sb = np.dot(Sw_inv, Sb)

    eig_vals, eig_vecs = np.linalg.eig(Sw_inv_Sb)
    eig_pairs = [(eig_vals[i], eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_vals = np.real(eig_vals).reshape(1, 8)
    eig_pairs.sort()
    eig_pairs.reverse()

    p_mat = np.real(eig_pairs[0][1].reshape(1, 8))
    dims_reduced = len(p_mat)
    train_data = (np.delete(train_data, 8, 1)).transpose()
    train_data = np.dot(p_mat, train_data).transpose()
    train_data = np.column_stack((train_data, fixed)) #construct train_data for MLE classifier

    test_data = (np.delete(test_data, 8, 1)).transpose()
    test_data = np.dot(p_mat, test_data).transpose()
    test_data = np.column_stack((test_data, col_8_test)) #construct test_data for MLE classifier

    train_data_filter1 = train_data[:, 1] = train_data[:, 1] == 1 #setting filter for training data when column 9 = 1
    mean1 = train_data[train_data_filter1].mean(0) #calculate mean of each attribute
    mean1 = np.delete(mean1, 1, 0) #remove the 4th column from mean vector
    var1 = np.cov(train_data[train_data_filter1].transpose()) #calculate covariance matric
    var1 = np.delete(var1, 1, 1) #removes the 4th column from the covariance matrix
    var1 = np.delete(var1, 1, 0) #removes the 4th row from the covariance matrix

    train_data_filter0 = train_data[:, 1] = train_data[:, 1] == 0
    mean0 = train_data[train_data_filter0].mean(0)
    mean0 = np.delete(mean0, 1, 0)
    var0 = np.cov(train_data[train_data_filter0].transpose())
    var0 = np.delete(var0, 1, 1)
    var0 = np.delete(var0, 1, 0)

    priortmp1 = train_data[train_data_filter1].shape[0] #gives the count where training data column 4 equals 1
    priortmp0 = train_data[train_data_filter0].shape[0]

    prior0 = float(priortmp0) / (priortmp0 + priortmp1) #calculates prior probability of trainging data where column 4 equals 0
    prior1 = float(priortmp1) / (priortmp0 + priortmp1)

    constant1 = 1 / math.sqrt(np.linalg.det(var1)) #calculates 1/sqrt(covariance)
    constant0 = 1 / math.sqrt(np.linalg.det(var0))
    inv_var0 = np.linalg.inv(var0) #inverse of covariance matrix
    inv_var1 = np.linalg.inv(var1)

    for i in test_data:
        j = np.delete(i, 1, 0)
        diff0 = j - mean0
        lklhood0 = constant0 * np.exp(-np.dot(np.dot(diff0, inv_var0), diff0) / 2)

        diff1 = j - mean1
        lklhood1 = constant1 * np.exp(-np.dot(np.dot(diff1, inv_var1), diff1) / 2)

        post0 = prior0 * lklhood0
        post1 = prior1 * lklhood1

        if(post0 > post1 and i[1] == 0):
            correct += 1
        elif(post1 < post0 and i[1] == 1):
            correct += 1
        else:
            wrong += 1

    accuracy.append(100 * float(correct) / (correct + wrong))
    runs += 1

print "Average Accuracy: " + str(np.average(accuracy))
print "Standard Deviation of Accuracy: " + str(np.std(accuracy))
