import csv
import math
import numpy as np
import numpy.matlib

runs = 0
accuracy = []
#while (runs < 10):
#raw_data = np.genfromtxt("pima-indians-diabetes.csv", delimiter = ",", usecols = (0, 1, 2, 3, 4, 5, 6, 7, 8))
#    np.random.shuffle(raw_data) #randomize data
class1 = np.matrix([[-2, 1], [-5, -4], [-3, 1], [0, -3], [-8, -1]])
class2 = np.matrix([[2, 5], [1, 0], [5, -1], [-1, -3], [6, 1]])

mu1 = class1.mean(0)
mu2 = class2.mean(0)
print "Class 1 Mean: "
print mu1
print "\nClass 2 Mean: "
print mu2

scatter1 = (len(class1) - 1) * np.cov(class1.transpose())
scatter2 = (len(class2) - 1) * np.cov(class2.transpose())
scatter_w = scatter1 + scatter2
inv_scatter_w = np.linalg.inv(scatter_w)
v = np.dot(inv_scatter_w, (mu1.transpose() - mu2.transpose()))
y1 = np.dot(v.transpose(), class1.transpose())
y2 = np.dot(v.transpose(), class2.transpose())
print "\nClass 1 Scatter: "
print scatter1
print "\nClass 2 Scatter: "
print scatter2
print "\nWithin Class Scatter: "
print inv_scatter_w
print "\nOptimal Line Direction: "
print v
print "\nClass 1 Projection: "
print y1
print "\nClass 2 Projection: "
print y2
print "\nAll points were classified correctly except for [-1, -3]\n"
