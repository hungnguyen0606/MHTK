import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def getInfo(X):
    #print X
    try:

        mean = np.mean(X, axis = 0)
        assert mean.shape[0] == 2, 'Mean err'
        std = np.std(X, axis = 0)
        assert std.shape[0] == 2, 'Std err'
        minn = np.min(X, axis = 0)
        maxx = np.max(X, axis = 0)
        assert minn.shape[0] == 2 and maxx.shape[0] == 2, 'Min or Max err'

        print 'Mean X: %f.\nStd X: %f'%(mean[0], std[0])
        print 'Min X: %f.\nMax X: %f\nRange X: %f'%(minn[0], maxx[0], maxx[0] - minn[0])
        print ''
        print 'Mean Y: %f.\nStd Y: %f'%(mean[1], std[1])
        print 'Min Y: %f.\nMax Y: %f\nRange Y: %f'%(minn[1], maxx[1], maxx[1] - minn[1])
        #-------------------------------------------------------------------
        x1 = np.linspace(0.0, 5.0)
        x2 = np.linspace(0.0, 2.0)

        y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
        y2 = np.cos(2 * np.pi * x2)

        # plt.subplot(2, 1, 1)
        # plt.plot(x1, y1, 'ko-')
        # plt.title('A tale of 2 subplots')
        # plt.ylabel('Damped oscillation')
        # plt.xlabel('Hello')
        # plt.subplot(2, 1, 2)
        # plt.plot(x2, y2, 'r.-')
        # plt.xlabel('time (s)')
        # plt.ylabel('Undamped')

        # plt.show()
        plt.subplot(2,1,1)
        plt.hist(X[:, 0], bins = 20, color = 'b')
        plt.ylabel('Count')
        plt.xlabel('X')

        plt.subplot(2,1,2)
        plt.hist(X[:, 1], bins = 20, color = 'r')
        plt.ylabel('Count')
        plt.xlabel('Y')
        plt.show()
    except Exception as err:
        print err
    finally:
        pass

def main():
    #generateData(200, 'input200.txt')
    N, x, z = loadData('input200.txt')
    #getInfo(x)

    
    listInput = ['input200.txt']
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

            print svm_acc
            print reg_acc
            print np.mean(svm_acc)*100, np.mean(reg_acc)*100
            print np.std(svm_acc), np.std(reg_acc)
    except Exception as err:
        print err
    finally:
        pass

if __name__ == '__main__':
        main()

