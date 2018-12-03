import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import OneClassSVM 


class roc:

	def __init__(self, y):
		self.y = y
		self.data = [None, None, None] * 0
		self.info = [['r', 'Linear'], ['g', '2ndSvm'], ['b', 'Gradient']]
		
	def append(self, y, model):
		self.data.append([y, self.info[model][0], self.info[model][1]])

	def plot(self):
	#labels and title setting
		for i in range(len(self.data)):
			tn,fp,fn,tp = confusion_matrix(self.y, self.data[i][0]).ravel()
			specificity = float(tn) / (tn+fp)
			sensitivity = float(tp) / (tp+fn)
			plt.plot(specificity,sensitivity, marker = 'o', color = self.data[i][1], label = self.data[i][2])

		plt.title('ROC')
		plt.xlabel("Specificity")
		plt.ylabel("Sensitivity")
		plt.legend(loc='lower left')

	#graph saving as .png file
		#plt.figure()
		plt.savefig("Results//testroc.png")
		plt.show()

if __name__ == '__main__':
	prefix = str("dataWtype//")
	List_x = [None] * 8
	List_y = [None] * 8
	y_real = np.zeros((0))
	for i in range(8):
		x_name = prefix + "X" + str(i) + ".npy"
		y_name = prefix + "Y" + str(i) + ".npy"

		List_x[i] = np.load(x_name)
		List_y[i] = np.load(y_name)

		y_real = np.concatenate((y_real, List_y[i]), axis = None)

	data = roc(y_real)

	linear = [[0.9, 1],[0.5, 1],[0.3, 1],[0.7, 1],[0.8, 1],[0.6, 1],[0.5, 1],[0.6, 1],[1, 1],[0.6, 1]]
	secsvm = [[0.8, 0],[0.5, 0],[0.5, 0],[0.7, 0],[0.2, 1],[0.8, 0],[0.6, 0],[0.7, 1],[0.3, 0],[0.9, 0]]
	gradie = [[0.03, 220, 1],[0.03, 200, 0],[0.03, 210, 0],[0.03, 180, 0],[0.02, 220, 1],[0.02, 160, 0],[0.05, 190, 1],[0.03, 160, 0],[0.02, 220, 0],[0.02, 220, 1]]
	
	result = np.zeros((0))
	n, d = List_x[0].shape
	for i in range(8):
            test_x = List_x[i]
            test_y = List_y[i]
            x = np.zeros((0, d))
            y = np.zeros((0))
            for j in range(8):
                if not i == j:
                    x = np.concatenate((x, List_x[j]), axis = 0)
                    y = np.concatenate((y, List_y[j]), axis = None)

            clf = svm.LinearSVC(penalty = 'l2', loss = 'squared_hinge', dual = False, C = linear[i][0])
            x = [x, x[:, 37:44]][linear[i][1]]
            test_x = [test_x, test_x[:, 37:44]][linear[i][1]]
            clf.fit(x,y)
            result = np.concatenate((result, clf.predict(test_x)), axis = None)
	print(len(y_real[np.where(y_real == result)]) / 800.0)
	data.append(result, 0)

	result = np.zeros((0))
	for i in range(2):
		test_x = np.zeros((0, d))
		test_y = np.zeros((0))
		x = np.zeros((0, d))
		y = np.zeros((0))
		for j in range(4):
			x = np.concatenate((x, List_x[(j + i * 4 + 4) % 8]), axis = 0)
			y = np.concatenate((y, List_y[(j + i * 4 + 4) % 8]), axis = None)
            
		for j in range(4):
			test_x = np.concatenate((test_x, List_x[(j + i * 4) % 8]), axis = 0)
			test_y = np.concatenate((test_y, List_y[(j + i * 4) % 8]), axis = None)

		clf = svm.LinearSVC(penalty = 'l2', loss = 'squared_hinge', dual = False, C = linear[i + 8][0])
		x = [x, x[:, 37:44]][linear[i + 8][1]]
		test_x = [test_x, test_x[:, 37:44]][linear[i + 8][1]]
		clf.fit(x,y)
		result = np.concatenate((result, clf.predict(test_x)), axis = None)	
	#print(len(result))
	print(len(y_real[np.where(y_real == result)]) / 800.0)
	data.append(result, 0)

	result = np.zeros((0))
	for i in range(8):
            test_x = List_x[i]
            test_y = List_y[i]
            x = np.zeros((0, d))
            y = np.zeros((0))
            for j in range(8):
                if not i == j:
                    x = np.concatenate((x, List_x[j]), axis = 0)
                    y = np.concatenate((y, List_y[j]), axis = None)

            clf = svm.SVC(kernel = 'poly', degree = 2, C = secsvm[i][0], gamma = 'scale')
            x = [x, x[:, 37:44]][secsvm[i][1]]
            test_x = [test_x, test_x[:, 37:44]][secsvm[i][1]]
            clf.fit(x,y)
            result = np.concatenate((result, clf.predict(test_x)), axis = None)
	#print(len(result))
	print(len(y_real[np.where(y_real == result)]) / 800.0)
	data.append(result, 1)

	result = np.zeros((0))
	for i in range(2):
		test_x = np.zeros((0, d))
		test_y = np.zeros((0))
		x = np.zeros((0, d))
		y = np.zeros((0))
		for j in range(4):
			x = np.concatenate((x, List_x[(j + i * 4 + 4) % 8]), axis = 0)
			y = np.concatenate((y, List_y[(j + i * 4 + 4) % 8]), axis = None)
            
		for j in range(4):
			test_x = np.concatenate((test_x, List_x[(j + i * 4) % 8]), axis = 0)
			test_y = np.concatenate((test_y, List_y[(j + i * 4) % 8]), axis = None)

		clf = svm.SVC(kernel = 'poly', degree = 2, C = secsvm[i + 8][0], gamma = 'scale')
		x = [x, x[:, 37:44]][secsvm[i + 8][1]]
		test_x = [test_x, test_x[:, 37:44]][secsvm[i + 8][1]]
		clf.fit(x,y)
		result = np.concatenate((result, clf.predict(test_x)), axis = None)	
	#print(len(result))
	print(len(y_real[np.where(y_real == result)]) / 800.0)
	data.append(result, 1)

	result = np.zeros((0))
	for i in range(8):
			test_x = List_x[i]
			test_y = List_y[i]
			x = np.zeros((0, d))
			y = np.zeros((0))
			for j in range(8):
				if not i == j:
					x = np.concatenate((x, List_x[j]), axis = 0)
					y = np.concatenate((y, List_y[j]), axis = None)

			clf = GradientBoostingClassifier(learning_rate = gradie[i][0], n_estimators = int(gradie[i][1]), max_depth=3)
			x = [x, x[:, 37:44]][gradie[i][2]]
			test_x = [test_x, test_x[:, 37:44]][gradie[i][2]]
			clf.fit(x,y)
			result = np.concatenate((result, clf.predict(test_x)), axis = None)
	#print(len(result))
	print(len(y_real[np.where(y_real == result)]) / 800.0)
	data.append(result, 2)

	result = np.zeros((0))
	for i in range(2):
		test_x = np.zeros((0, d))
		test_y = np.zeros((0))
		x = np.zeros((0, d))
		y = np.zeros((0))
		for j in range(4):
			x = np.concatenate((x, List_x[(j + i * 4 + 4) % 8]), axis = 0)
			y = np.concatenate((y, List_y[(j + i * 4 + 4) % 8]), axis = None)
            
		for j in range(4):
			test_x = np.concatenate((test_x, List_x[(j + i * 4) % 8]), axis = 0)
			test_y = np.concatenate((test_y, List_y[(j + i * 4) % 8]), axis = None)

		clf = GradientBoostingClassifier(learning_rate = gradie[i + 8][0], n_estimators = int(gradie[i + 8][1]), max_depth=3)
		x = [x, x[:, 37:44]][gradie[i + 8][2]]
		test_x = [test_x, test_x[:, 37:44]][gradie[i + 8][2]]
		clf.fit(x,y)
		result = np.concatenate((result, clf.predict(test_x)), axis = None)	
	#print(len(result))
	print(len(y_real[np.where(y_real == result)]) / 800.0)
	data.append(result, 2)

	data.plot()