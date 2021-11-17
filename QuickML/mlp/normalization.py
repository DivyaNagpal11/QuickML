class Normalization:
    def __init__(self, data_frame):
        self._df=data_frame

    def normalization_methods(self, table_data):
        for each in table_data:
            feature = each['feature']
            method = each['method']
            if method == "Normalization in range":
                low = float(each['lower'])
                high = float(each['higher'])
                minimum = self._df[feature].min()
                maximum = self._df[feature].max()
                self._df[feature] = self._df[feature].apply(normalization_in_range, args=(low, high, minimum, maximum))
            else:
                mean = self._df[feature].mean()
                standard_deviation = self._df[feature].std()
                self._df[feature] = self._df[feature].apply(standard_normal_distribution, args=(mean, standard_deviation))
        return self._df


def normalization_in_range(x, low, high, minimum, maximum):
    result = ((((x-minimum)/(maximum-minimum))*(high-low))+low)
    return result


def standard_normal_distribution(x, mean, standard_deviation):
    result = (x-mean)/standard_deviation
    return result
