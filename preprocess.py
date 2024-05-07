import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from io import StringIO
import requests


class PreProcess:
    
    def __init__(self, file_url, train_split_size, batch_size, seq_length):
        self.file_url = file_url
        self.train_split_size = train_split_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        # data:
        self.data = None    # original dataset
        self.dates = None   # original dates
        self.dates_train = None
        self.dates_test = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_seq = None
        self.y_train_seq = None
        self.X_train_batches = None
        self.y_train_batches = None

    def load_data(self):
        res = requests.get(self.file_url)
        if res.ok:
            csv_data = res.text
            df = pd.read_csv(StringIO(csv_data))  # reads CSV data from string
            self.data = pd.DataFrame(df)
        else:
            print(f"Failed to retrieve data: {res.status_code} - {res.reason}")

    def handle_missing_values(self):
        # remove rows with missing labels and reset indices
        df = self.data.copy()
        df.drop_duplicates(inplace=True)
        df = df.dropna(subset=[df.columns[-1]]).reset_index(drop=True)

        # load features and labels
        self.X = pd.DataFrame(df.iloc[:, 1:-2].values)
        self.y = pd.DataFrame(df.iloc[:, -1].values)
        self.dates = pd.DataFrame(df.iloc[:, :1].values) # store dates separately

        # replace missing X values (numerical) using the median along each column
        imputer = SimpleImputer(strategy='median')
        self.X = pd.DataFrame(imputer.fit_transform(self.X), columns=self.X.columns, index=self.X.index)

    def split_data(self):
        # splits data into training and testing sets, maintaining sequential order
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = self.train_split_size, shuffle=False)
        self.dates_train, self.dates_test = train_test_split(self.dates, train_size = self.train_split_size, shuffle=False)
    
    def normalize(self):
        # rescales training and test data to [0, 1]
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.fit_transform(self.X_test)
        self.y_train = self.scaler.fit_transform(self.y_train)
        self.y_test = self.scaler.fit_transform(self.y_test)

    def denormalize(self, norm_data):
        og_shape = np.array(norm_data).shape
        norm_data_2d = np.reshape(norm_data, (-1, 1))  # convert to 2d array format
        denorm_data = self.scaler.inverse_transform(norm_data_2d)
        denorm_data = np.reshape(denorm_data, og_shape) # convert back to og shape
        print(denorm_data)
        return denorm_data  # np array

    def create_sequences(self):
        # iterates over training data using sliding window approach
        X_seq = []
        y_seq = []
        for i in range(len(self.X_train) - self.seq_length):
            X_seq.append(self.X_train[i:i+self.seq_length])
            y_seq.append(self.y_train[i + self.seq_length])
        self.X_train_seq = np.array(X_seq)
        self.y_train_seq = np.array(y_seq)
    
    def create_batches(self):
        '''
        Shape of X_train_batches: (num_batches, batch_size, seq_length, num_features)
        :return:
        '''
        # creates batches of the training sequences
        num_batches = len(self.X_train_seq) // self.batch_size
        self.X_train_batches = np.array_split(self.X_train_seq[:num_batches * self.batch_size], num_batches)
        self.y_train_batches = np.array_split(self.y_train_seq[:num_batches * self.batch_size], num_batches)

    def preprocess_data(self):
        self.load_data()
        self.handle_missing_values()
        self.split_data()
        self.normalize()
        self.create_sequences()
        self.create_batches()