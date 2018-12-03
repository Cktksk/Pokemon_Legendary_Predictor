import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder

def generateData(data_pos, data_neg, filename):
    n, d = data_pos.shape
    d = d - 1
    for i in range(8):
        namex = filename + "X" + str(i) + ".npy"
        namey = filename + "Y" + str(i) + ".npy"
        x = np.zeros((100,d))
        y = np.zeros((100))

        if i < 7:
            x[0:92] = data_pos[(i*92):(i*92+92), 0:d]
            y[0:92] = data_pos[(i*92):(i*92+92), d]
            x[92:100] = data_neg[(i*8):(i*8+8), 0:d]
            y[92:100] = data_neg[(i*8):(i*8+8), d]

        else:
            x[0:91] = data_pos[644:735, 0:d]
            y[0:91] = data_pos[644:735, d]
            x[91:100] = data_neg[56:65, 0:d]
            y[91:100] = data_neg[56:65, d]

        np.save(namex,x)
        np.save(namey,y)
    


if __name__ == '__main__':
    # Read csv and drop unnecessary data
    raw_csv = pd.read_csv("Pokemon.csv")
    types = raw_csv[['Type 1', 'Type 2']]
    raw_csv = raw_csv.drop(['#','Type 1', 'Type 2','Total','Generation','Name'], axis = 1)
    
    # Generate type code
    types = types.values.astype(str)
    enc = OneHotEncoder(handle_unknown = 'ignore')
    enc.fit(types)
    cleaned_type = enc.transform(types).toarray()

    data = raw_csv.values
    data = np.concatenate((cleaned_type, data),axis = 1)
    true_columns = list(np.where(data[:,43] == True))
    false_columns = list(np.where(data[:,43] == False))
    data[true_columns, 43] = -1
    data[false_columns, 43] = 1
    data_pos = data[np.where(data[:,43] == 1)]
    data_neg = data[np.where(data[:,43] == -1)]

    # Randomize data
    np.random.shuffle(data_pos)
    np.random.shuffle(data_pos)
    np.random.shuffle(data_neg)
    np.random.shuffle(data_neg)

    # Generate data table
    np.save("dataWtype//data.npy", data)
    np.save("dataWOtype//data.npy", data[:, 37:44])

    # Split data
    generateData(data_pos, data_neg, "dataWtype//")
    generateData(data_pos[:, 37:], data_neg[:, 37:], "dataWOtype//")