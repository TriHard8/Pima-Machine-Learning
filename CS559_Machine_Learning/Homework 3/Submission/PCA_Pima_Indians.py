import csv
import math
import numpy as np
import numpy.matlib

problem1 = open('Problem1.txt', 'w')
reduced_dims = 3
runs = 0
accuracy = []
while (runs < 10):
    raw_data = np.genfromtxt("pima-indians-diabetes.csv", delimiter = ",", usecols = (0, 1, 2, 3, 4, 5, 6, 7, 8))
    np.random.shuffle(raw_data) #randomize data

    train_data = raw_data[0:len(raw_data)/2] #assign 1st half of data to training data
    test_data = raw_data[len(raw_data)/2:] #assign 2nd half of data to test data
    correct = 0
    wrong = 0
    pca_train_data = train_data[:,0:8].transpose()
    col_8 = train_data[:, 8]
    col_8_test = test_data[:, 8]
    test_data = test_data[:, 0:8].transpose()
    pca_train_mean = np.matrix(pca_train_data.mean(1)) #calculate training mean
    pca_train_mean0 = pca_train_data - np.matlib.repmat(pca_train_mean.transpose(), 1, len(train_data)) #training data with the means subtracted
    pca_train_cov = np.cov(pca_train_mean0)
    pca_eig_vals, pca_eig_vecs = np.linalg.eig(pca_train_cov) #calculates eigenvalues/vectors from the training covariance matrix
    print pca_eig_vecs
    eig_pairs = [(pca_eig_vals[i], pca_eig_vecs[:,i]) for i in range(len(pca_eig_vals))]

    eig_pairs.sort()
    eig_pairs.reverse()
    
    p_mat = np.hstack((eig_pairs[0][1].reshape(8, 1),
                       eig_pairs[1][1].reshape(8, 1),
                       eig_pairs[2][1].reshape(8, 1))) #stacks the eigenvectors corresponding to the top 3 eigenvalues
    p_mat = p_mat.transpose()
    train_data = np.dot(p_mat, pca_train_data).transpose()
    train_data = np.column_stack((train_data, col_8)) #adds classification column back to training matrix for MLE classification
    test_data = np.dot(p_mat, test_data).transpose()
    test_data = np.column_stack((test_data, col_8_test)) #adds classification column back to testing matrix for MLE classification

    train_data_filter1 = train_data[:, 3] = train_data[:, 3] == 1 #setting filter for training data when column 9 = 1
    mean1 = train_data[train_data_filter1].mean(0) #calculate mean of each attribute
    mean1 = np.delete(mean1, 3, 0) #remove the 4th column from mean vector
    var1 = np.cov(train_data[train_data_filter1].transpose()) #calculate covariance matric
    var1 = np.delete(var1, 3, 1) #removes the 4th column from the covariance matrix
    var1 = np.delete(var1, 3, 0) #removes the 4th row from the covariance matrix

    train_data_filter0 = train_data[:, 3] = train_data[:, 3] == 0
    mean0 = train_data[train_data_filter0].mean(0)
    mean0 = np.delete(mean0, 3, 0)
    var0 = np.cov(train_data[train_data_filter0].transpose())
    var0 = np.delete(var0, 3, 1)
    var0 = np.delete(var0, 3, 0)

    priortmp1 = train_data[train_data_filter1].shape[0] #gives the count where training data column 4 equals 1
    priortmp0 = train_data[train_data_filter0].shape[0]

    prior0 = float(priortmp0) / (priortmp0 + priortmp1) #calculates prior probability of trainging data where column 4 equals 0
    prior1 = float(priortmp1) / (priortmp0 + priortmp1)

    constant1 = 1 / math.sqrt(np.linalg.det(var1)) #calculates 1/sqrt(covariance)
    constant0 = 1 / math.sqrt(np.linalg.det(var0))
    inv_var0 = np.linalg.inv(var0) #inverse of covariance matrix
    inv_var1 = np.linalg.inv(var1)

    for i in test_data:
        j = np.delete(i, 3, 0)
        diff0 = j - mean0
        lklhood0 = constant0 * np.exp(-np.dot(np.dot(diff0, inv_var0), diff0) / 2)

        diff1 = j - mean1
        lklhood1 = constant1 * np.exp(-np.dot(np.dot(diff1, inv_var1), diff1) / 2)

        post0 = prior0 * lklhood0
        post1 = prior1 * lklhood1

        if(post0 > post1 and i[3] == 0):
            correct += 1
        elif(post1 < post0 and i[3] == 1):
            correct += 1
        else:
            wrong += 1

    accuracy.append(100 * float(correct) / (correct + wrong))
    runs += 1

print "Average Accuracy: " + str(np.average(accuracy))
print "Standard Deviation of Accuracy: " + str(np.std(accuracy))

