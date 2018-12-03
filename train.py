import numpy as np
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import OneClassSVM 

# args
# whatever x
# whatever y
# modelType has to be 0,1,2
#   modelType = 0 : gradient
#   modelType = 1 : LinearSVM
#   modelType = 2 : 2ndKernelSVM

class train:
    def __init__(self, x, y, modelType):
        self.C = np.array(range(10)) / 10.0 + 0.1
        self.rate = np.array(range(5)) / 100.0 + 0.01
        self.n = np.array(range(10)) * 10.0 + 150
        self.model = None
        x_pos = x[np.where(y == 1)]
        x_neg = x[np.where(y == -1)]
        y_pos = y[np.where(y == 1)]
        y_neg = y[np.where(y == -1)]
        self.x_pos = [x_pos, x_pos[:, 37:44]]
        self.x_neg = [x_neg, x_neg[:, 37:44]]
        self.y_pos = y_pos
        self.y_neg = y_neg
        self.n_pos = len(x_pos)
        self.n_neg = len(x_neg)
        self.modelType = modelType
        self.feature = None

    def booststrapping(self, i, model):
        acc = np.zeros(30)
        for b in range(30):
            train_samples_pos = list(np.random.randint(0, self.n_pos, self.n_pos))
            test_samples_pos = list(set(range(self.n_pos)) - set(train_samples_pos))
            train_samples_neg = list(np.random.randint(0, self.n_neg, self.n_neg))
            test_samples_neg = list(set(range(self.n_neg)) - set(train_samples_neg))
            train_x = np.concatenate((self.x_pos[i][train_samples_pos], self.x_neg[i][train_samples_neg]), axis = 0)
            train_y = np.concatenate((self.y_pos[train_samples_pos], self.y_neg[train_samples_neg]), axis = 0)
            test_x = np.concatenate((self.x_pos[i][test_samples_pos], self.x_neg[i][test_samples_neg]), axis = 0)
            test_y = np.concatenate((self.y_pos[test_samples_pos], self.y_neg[test_samples_neg]), axis = 0)
            model.fit(train_x, train_y)
            acc[b] = np.mean(test_y == model.predict(test_x))
        
        return np.mean(acc) 

    def training(self):
        paraFunc = self.modelfuncs[self.modelType]
        paraFunc(self)
        return self.model, self.feature
       
    def gradientPara(self):
        bs_acc = 0
        print("GradientBoostingClassifier")
        for i in range(2):
            for j in range(5):
                for k in range(10):
                    model = GradientBoostingClassifier(learning_rate = self.rate[j], n_estimators = int(self.n[k]), max_depth=3)

                    acc = self.booststrapping(i, model)

                    if acc > bs_acc:
                        bs_acc = acc
                        self.model = model
                        self.feature = i

        print("best para: " + str(self.model.get_params()))
        print("best acc: " + str(bs_acc))
           
    def linearPara(self):
        bs_acc = 0
        print("LinearSVM")
        for i in range(2):
            for j in range(10):
                model = svm.LinearSVC(penalty = 'l2', loss = 'squared_hinge', dual = False, C = self.C[j])

                acc = self.booststrapping(i, model)

                if acc > bs_acc:
                        bs_acc = acc
                        self.model = model
                        self.feature = i

        print("best para: " + str(self.model.get_params()))
        print("best acc: " + str(bs_acc))

    def kerPara(self):
        bs_acc = 0
        print("2ndSVM")
        for i in range(2):
            for j in range(10):
                model = svm.SVC(kernel = 'poly', degree = 2, C = self.C[j], gamma = 'scale')

                acc = self.booststrapping(i, model)

                if acc > bs_acc:
                        bs_acc = acc
                        self.model = model
                        self.feature = i

        print("best para: " + str(self.model.get_params()))
        print("best acc: " + str(bs_acc))

    modelfuncs = [linearPara, kerPara, gradientPara]