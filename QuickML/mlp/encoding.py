import pandas as pd


class Encoder:
    def __init__(self, data_frame):
        self._data_frame = data_frame

    def label_encoding(self,selected_features):
        for feature in selected_features:
            self._data_frame[feature] = self._data_frame[feature].astype('category')
            self._data_frame[feature] = self._data_frame[feature].cat.codes
        return self._data_frame

    def one_hot_encoding(self,selected_features):
        data_frame = pd.get_dummies(self._data_frame, columns=selected_features)
        return data_frame
