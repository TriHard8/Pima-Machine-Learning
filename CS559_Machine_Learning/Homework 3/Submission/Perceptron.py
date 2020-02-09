import numpy as np

def all_bools_true(lst): #test if all items of list are true
    if(lst[0] == True):
        return all(lst) or not any(lst)
    else:
        return False

X = np.array([  [2, 1, 1, -1, 0, 2],
                [1, 0, 0, 1, 2, 0],
                [2, -1, -1, 1, 1, 0],
                [1, 4, 0, 1, 2, 1],
                [1, -1, 1, 1, 1, 0],
                [1, -1, -1, -1, 1, 0],
                [2, -1, 1, 1, 2, 1]])

len_X = len(X)
X.reshape(1, 42)
classified = [False, False, False, False, False, False, False] #list to contain true if classified correctly and false if misclassified

for i in range(0, len(X)):
    X = np.insert(X, 1 +i*len_X, 1) #inserts 1 into data set

X = X.reshape(1, 49)

for i in range(0, len_X):
    if(X[0][7*i] == 2):
         for j in range(1, 7):
             X[0][7*i+j] *= -1 #negates anything belonging to class 2.

a = [3, 1, 1, -1, 2, 7]
i = 0
while(not all_bools_true(classified)):
    test = 0
    for j in range(0, 6):
        test += a[j] * X[0][7*i+j+1]  #test value accumulation
    
    if(test > 0): #if text <= 0 then misclassified
        classified[i] = True
    else:
        for j in range(0, 6):
            a[j] += X[0][7*i+j+1] #change a matrix
            classified[i] = False #if `a` vector has to be modified, then set all data points to misclassified so we make sure to test them all with the new 'a' vector.
    
    i = (i + 1) % 7

print a

