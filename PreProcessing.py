import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PreProcessing:
    def __init__(self, target_value: str = 'class',
                 scaler=StandardScaler(),
                 verbose: bool = True
                 ):
        self.target_value = target_value
        self.scaler = scaler
        self.verbose = verbose

    def transform(self, X: pd.DataFrame):
        X = X.astype('float32')
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame):
        if self.verbose and self.scaler is not None:
            print("Transforming and fitting data")
        X = X.astype('float32')
        if self.scaler is not None:
            return self.scaler.fit_transform(X)
        return X

    def split_data_in_train_test(self, X, y, test_ratio: float = 0.15):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y)

        if self.verbose:
            print('shape of the training set of X', X_train.shape)
            print('shape of the test set of X', X_test.shape)
        return X_train, X_test, y_train, y_test

    def get_feature_target_dataframe(self, df: pd.DataFrame):
        y = df[self.target_value]
        X = df.drop([self.target_value], axis=1)
        return X, y
