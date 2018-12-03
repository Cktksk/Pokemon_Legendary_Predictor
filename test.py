import sys
import numpy as np
from train import train

def run():
    prefix = str("dataWtype//")
    List_x = [None] * 8
    List_y = [None] * 8
    for i in range(8):
        x_name = prefix + "X" + str(i) + ".npy"
        y_name = prefix + "Y" + str(i) + ".npy"

        List_x[i] = np.load(x_name)
        List_y[i] = np.load(y_name)
    
    op = open("res.txt", "w")

    n, d = List_x[0].shape
    for k in range(3):
        if k == 0:
            op.write("linear\n")
        elif k == 1:
            op.write("2ndker\n")
        else:
            op.write("gradient\n")

        acc = np.zeros(8)
        feature = None
        for i in range(8):
            test_x = List_x[i]
            test_y = List_y[i]
            x = np.zeros((0, d))
            y = np.zeros((0))
            for j in range(8):
                if not i == j:
                    x = np.concatenate((x, List_x[j]), axis = 0)
                    y = np.concatenate((y, List_y[j]), axis = None)

            clf, feature = train(x, y, k).training()
            x = [x, x[:, 37:44]]
            test_x = [test_x, test_x[:, 37:44]]
            clf.fit(x[feature],y)
            result = clf.predict(test_x[feature])
            acc[i] = np.mean(test_y == result)
            print(feature)
            print("test: " + str(acc[i]))

        print(np.mean(acc))
        op.write("8 fold: " + str(np.mean(acc)) + "\n")

        acc = np.zeros(2)
        for i in range(2):
            test_x = np.zeros((0, d))
            test_y = np.zeros((0))
            x = np.zeros((0, d))
            y = np.zeros((0))
            for j in range(4):
                x = np.concatenate((x, List_x[(j + i * 4) % 8]), axis = 0)
                y = np.concatenate((y, List_y[(j + i * 4) % 8]), axis = None)
            
            for j in range(4):
                test_x = np.concatenate((test_x, List_x[(j + i * 4 + 4) % 8]), axis = 0)
                test_y = np.concatenate((test_y, List_y[(j + i * 4 + 4) % 8]), axis = None)

            clf, feature = train(x, y, k).training()
            x = [x, x[:, 37:44]]
            test_x = [test_x, test_x[:, 37:44]]
            clf.fit(x[feature],y)
            result = clf.predict(test_x[feature])
            acc[i] = np.mean(test_y == result)
            print(feature)
            print("test: " + str(acc[i]))
            
        print(np.mean(acc))
        op.write("2 fold: " + str(np.mean(acc)) + "\n")
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    op.close()

if __name__ == '__main__':
    run()