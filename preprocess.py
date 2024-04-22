'''
This file will contain the pre_process class and its methods
- normalize
- eliminate_outliers
- eliminate some columns/select the proper columns
- split into test and train
- etc
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sys


class PreProcess:
    def __init__(self, file_path, train_split_size):
        self.file_path = file_path
        self.train_split_size = train_split_size
        self.data = None    # original dataset
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def load_data(self):
        data = pd.read_csv(self.file_path)
        self.data = pd.DataFrame(data)


    def handle_missing_values(self):
        # remove rows with missing labels and reset indices
        df = self.data.copy()
        df.drop_duplicates(inplace=True)
        df = df.dropna(subset=[df.columns[-1]]).reset_index(drop=True)

        # load features and labels
        self.X = pd.DataFrame(df.iloc[:, :-1].values)
        self.y = pd.DataFrame(df.iloc[:, -1].values)

        # replace missing X values (numerical) using the median along each column
        imputer = SimpleImputer(strategy='median')
        self.X.iloc[:, 1:] = imputer.fit_transform(self.X.iloc[:, 1:])

        # fill in missing dates appropriately
        self.X[0] = self.X[0].replace("", None).ffill().bfill() # fills forward then backward


    def normalize(self, X_train, X_test, y_train, y_test):
        # rescales training and test data to [0, 1]
        scaler = MinMaxScaler()

        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.fit_transform(X_test)
        self.y_train = scaler.fit_transform(y_train)
        self.y_test = scaler.fit_transform(y_test)
      

    def eliminate_outliers(self):
        pass


    def transform_dates(self):
        # encodes dates sequentially, starting from 0...
        label_encoder = LabelEncoder()
        self.X[0] = pd.to_datetime(self.X[0], format='%d-%m-%Y')
        self.X[0] = label_encoder.fit_transform(self.X[0])


    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size = self.train_split_size)
        return X_train, X_test, y_train, y_test


def main():
    prepro = PreProcess(sys.argv[1], 0.8)
    prepro.load_data()
    prepro.handle_missing_values()
    prepro.transform_dates()
    X_train, X_test, y_train, y_test = prepro.split_data()
    prepro.normalize(X_train, X_test, y_train, y_test)

    print(prepro.X_train, "\n")
    print(prepro.X_test, "\n")
    print(prepro.y_train, "\n")
    print(prepro.y_test)


if __name__ == "__main__":
    main()