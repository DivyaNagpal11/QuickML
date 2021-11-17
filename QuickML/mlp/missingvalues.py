class MissingValues:
    def __init__(self, data_frame):
            self._data_frame = data_frame

    def missing_value_methods(self, tabledata):
        for each in tabledata:
            feature = each['feature']
            method = each['method']
            if method == '':
                continue
            elif method == "value":
                input_value = int(each['value'])
                self.value_method(feature, input_value)
            elif method == "mean":
                self.mean_method(feature)
            elif method == "median":
                self.median_method(feature)
            elif method == "mode":
                self.mode_method(feature)
            elif method == 'delete':
                self.delete_method(feature)
        return self._data_frame

    def value_method(self, feature, input_value):
        self._data_frame[feature].fillna(input_value, inplace=True)

    def mean_method(self, feature):
        self._data_frame[feature].fillna(self._data_frame[feature].mean(), inplace=True)

    def median_method(self, feature):
        self._data_frame[feature].fillna(self._data_frame[feature].median(), inplace=True)

    def mode_method(self, feature):
        self._data_frame[feature].fillna(self._data_frame[feature].mode()[0], inplace=True)

    def delete_method(self, feature):
        self._data_frame.dropna(subset=[feature], how='any', inplace=True)
