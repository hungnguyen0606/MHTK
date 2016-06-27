import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.cross_validation import KFold

a, b, c = 7, 2, 7
f = lambda x, y: int(a*x + b*y + c*x*y) % 3

def generateData(N, output = None):
    try:
        #uniform distribution
        X = np.random.random_sample(N)
        Y = np.random.random_sample(N)
        t = np.vectorize(f)
        z = t(X, Y)
        
        with open(output, 'w') as fo:
            fo.write(str(N) + '\n')
            for i in range(z.size):
                fo.write(str(X[i]) + ' ' + str(Y[i]) + ' ' + str(z[i]) + '\n')

    except Exception as err:
        print err

    finally:
        return (X, Y, z)

def loadData(input):
    try:
        with open(input, 'r') as fi:
            N = int(fi.readline())
            x = []
            z = []

            for i in range(N):
                line = fi.readline().split()

                assert len(line) == 3, "Line %d only has %d parameter(s)."%(i, len(line))

                x.append([float(line[0]), float(line[1])])
                z.append(int(line[2]))

            return (N, np.array(x), np.array(z))

    except Exception as err:
        print err
    finally:
        pass

def transformX(X, Y):
    try:
        z = [[X[i], Y[i]] for i in range(X.size)]
        return np.array(z)

    except Exception as err:
        print err

    finally:
        pass

def makeLinear(X):
    try:
        z = [[X[i][0], X[i][1], X[i][0]*X[i][1]] for i in range(X.shape[0])]
        return np.array(z)
    except Exception as err:
        print err

    finally:
        pass

def main():
    generateData(40000, 'input40000.txt')
    listInput = ['input200.txt', 'input2000.txt', 'input40000.txt']
    try:

        for i in range(len(listInput)):
            nInput = listInput[i]
            
            N, x, z = loadData(nInput)
            #create model
            svm_clf = svm.SVC(gamma = 250, decision_function_shape = 'ovo')
            regr = linear_model.LinearRegression()

            #generate Kfold index
            kf = KFold(N, 10)

            svm_acc = []
            reg_acc = []

            #training
            for train_id, test_id in kf:
                #train & test svm
                svm_clf.fit(x[train_id], z[train_id])
                nz = svm_clf.predict(x[test_id])
                svm_acc.append(np.sum(nz==z[test_id])*1.0/test_id.size)

                #train & test regression
                regr.fit(x[train_id], z[train_id])
                nz = regr.predict(x[test_id])
                nz = (nz+0.5).astype(int)

                reg_acc.append(np.sum(nz==z[test_id])*1.0/test_id.size)

            print np.mean(svm_acc)*100, np.mean(reg_acc)*100
    except Exception as err:
        print err
    finally:
        pass

if __name__ == '__main__':
        main()

