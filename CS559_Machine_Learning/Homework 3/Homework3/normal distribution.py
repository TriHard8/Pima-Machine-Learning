rows = 3
columns = 100

m = np.matrix([1, 2, 3])
diag = np.diag([10, 3, 1])
R = np.array([[0.6651, 0.7427, 0.0775],[0.7395, -0.6696, 0.0697], [0.1037, 0.0109, -0.9946]])
var3 = np.array([[49.15, 44.62, 6.870], [44.62, 58.64, 7.514], [6.87, 7.514, 2.067]])
var3_eig_val, var3_eig_vec = np.linalg.eig(var3)
print var3_eig_val
print var3_eig_vec
print "***********************************"

dat = np.random.normal(size = (rows,columns))
datmean = np.matrix(dat.mean(1))
datmean0 = dat - np.matlib.repmat(datmean.transpose(), 1, columns)
datcov = np.cov(datmean0)
#print dat
print datmean
print datcov
print "\n"
np.savetxt(problem1, datmean, fmt = '%.8g')
np.savetxt(problem1, datcov)
problem1.write('\n')



dat1 = dat + np.matlib.repmat(m.transpose(), 1, columns)
dat1mean = np.matrix(dat1.mean(1))
dat1mean0 = dat1 - np.matlib.repmat(dat1mean, 1, columns)
dat1cov = np.cov(dat1mean0)
#print "dat 1: *****"
#print dat1
print dat1mean
print dat1cov
print "\n"
np.savetxt(problem1, dat1mean, fmt = '%.8g')
np.savetxt(problem1, dat1cov)
problem1.write('\n')

dat2 = np.dot(diag, dat1)
dat2mean = np.matrix(dat2.mean(1))
dat2mean0 = dat2 - np.matlib.repmat(dat2mean, 1, columns)
#dat2mean0 = (dat2mean0 - dat2mean0.mean()) / np.std(dat2mean0)
dat2cov = np.cov(dat2mean0)
dat2_eig_vals, dat2_eig_vecs = np.linalg.eig(dat2cov)
#print "dat2: *****"
#print dat2
print dat2mean
print dat2cov
print dat2_eig_vals
print dat2_eig_vecs
print "\n"
np.savetxt(problem1, dat2mean, fmt = '%.8g')
np.savetxt(problem1, dat2cov)
problem1.write('\n')

dat3 = np.dot(R, dat2)
dat3mean = np.matrix(dat3.mean(1))
dat3mean0 = dat3 - np.matlib.repmat(dat3mean, 1, columns)
#dat3mean0 = (dat3mean0 - dat3mean0.mean()) / np.std(dat3mean0)
dat3cov = np.cov(dat3mean0)
dat3_eig_vals, dat3_eig_vecs = np.linalg.eig(dat3cov)
#print "dat3: *****"
#print dat3
print dat3mean
print dat3cov
print dat3_eig_vals
print dat3_eig_vecs
print "\n"
np.savetxt(problem1, dat3mean, fmt = '%.8g')
np.savetxt(problem1, dat3cov)
problem1.write('\n')
'''