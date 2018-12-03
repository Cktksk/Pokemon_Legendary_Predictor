# Pokemon Legendary Predictor

Documents:

    Pokemon.csv: Raw data

    output.npg: Data Pattern

Files:

    data:
        X0-X7.npy: each npy file is a (100,6) np 2d-array, corresponding to Y0-Y7
        Y0-Y7.npy: each npy file is a (100) np array, each contain 8-9 negative label, and 91-92 1abel
        All data is from Pokemon, no sample is repeated

    inputData:
        ...


Scripts:

    data.py:
        Generate npy file from x_clean.npy to npys in data

    cross.py:
        Input: sklearn.svm
        Output: training 7-1 4-4 acc

![][PokeLogo]



[PokeLogo]:https://github.com/Cktksk/MyCache/blob/master/ImageLogos/pokemon-logo-black-transparent.png
