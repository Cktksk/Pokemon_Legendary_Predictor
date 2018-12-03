import sys
import numpy as np
from itertools import combinations
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import OneClassSVM 

def run(clf, data, clfid):
    #clf = svm.LinearSVC(penalty = 'l2', loss = 'squared_hinge', dual = False, C = 1)
    reportFile = open("./Results/result.txt", "a")

    if clfid == 0:
        reportFile.write("gradient boost classifier: \n")
        print("gradient boost classifier: ")
    elif clfid == 1:
        reportFile.write("linear SVC: \n")
        print("linear SVC: ")
    elif clfid == 2:
        reportFile.write("2nd degree polynomial: \n")
        print("2nd degree polynomial: ")
    elif clfid == 3:
        reportFile.write("3rd degree polynomial: \n")
        print("3rd degree polynomial:")
    elif clfid == 4:
        reportFile.write("one Class SVM: \n")
        print("one Class SVM:")

    prefix = str()
    if data == 1 :
        prefix = str("dataWOtype//")
        #print("Without Type")
    else:
        prefix = str("dataWtype//")
        #print("With Type")
    #print(clf.get_params())
    ## read X0-X7, Y0-Y7
    List_x = [None] * 8
    List_y = [None] * 8
    for i in range(8):
        x_name = prefix + "X" + str(i) + ".npy"
        y_name = prefix + "Y" + str(i) + ".npy"

        List_x[i] = np.load(x_name)
        List_y[i] = np.load(y_name)

    n, d = List_x[0].shape
    ## training acc
    count = 0
    x = np.zeros((800, d))
    y = np.zeros((800))
    for i in range(8):
        x[i*100:i*100+100] = List_x[i]
        y[i*100:i*100+100] = List_y[i]

    clf.fit(x,y)
    result = clf.predict(x)
    count = len(y[np.where(y == result)])
    
    reportFile.write("8-8 Training acc: {}\n".format(str(count / 800.0)))
    print("Training acc: " + str(count / 800.0))

    ## 7-1 test acc
    count = 0
    for i in range(8):
        test_x = List_x[i]
        test_y = List_y[i]
        x = np.zeros((0, d))
        y = np.zeros((0))
        for j in range(8):
            if not i == j:
                x = np.concatenate((x, List_x[j]), axis = 0)
                y = np.concatenate((y, List_y[j]), axis = None)

        clf.fit(x,y)
        result = clf.predict(test_x)
        for j in range(100):
            if test_y[j] == result[j]:
                count = count + 1
    reportFile.write("7-1 test acc: {}\n".format(str(count / 800.0)))
    print("7-1 test acc: " + str(count / 800.0))

    ## 4-4 test acc
    count = 0
    # train the first 400 data
    x = np.zeros((400, d))
    y = np.zeros((400))
    for j in range(4):
        x[j*100:j*100+100] = List_x[j]
        y[j*100:j*100+100] = List_y[j]

    test_x = np.zeros((400,d))
    test_y = np.zeros((400))

    for j in range(4,8):
        test_x[(j-4)*100:(j-4)*100+100] = List_x[j]
        test_y[(j-4)*100:(j-4)*100+100] = List_y[j]

    clf.fit(x,y)
    result = clf.predict(test_x)
    count = len(y[np.where(test_y == result)])

    # train the last 400 data
    for j in range(4,8):
        x[(j-4)*100:(j-4)*100+100] = List_x[j]
        y[(j-4)*100:(j-4)*100+100] = List_y[j]

    test_x = np.zeros((400,d))
    test_y = np.zeros((400))

    for j in range(4):
        test_x[j*100:j*100+100] = List_x[j]
        test_y[j*100:j*100+100] = List_y[j]

    clf.fit(x,y)
    result = clf.predict(test_x)
    count += len(y[np.where(test_y == result)])

    reportFile.write("4-4 test acc: {}\n".format(str(count / 800.0)))
    print("4-4 test acc: " + str(count / 800.0))
    reportFile.close()
    return

if __name__ == '__main__':
    algo = [GradientBoostingClassifier(learning_rate= 0.02, n_estimators=180, max_depth=3), 
    svm.LinearSVC(penalty = 'l2', loss = 'squared_hinge', dual = False, C = 0.5), 
    svm.SVC(kernel = 'poly', degree = 2, C = 0.5, gamma = 'scale'), 
    svm.SVC(kernel = 'poly', degree = 3, C = 0.5, gamma = 'scale'), 
    OneClassSVM(gamma = 0.5)]

    for i in range(len(algo)):
        run(algo[i], 1, i)
