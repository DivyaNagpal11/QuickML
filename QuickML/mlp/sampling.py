import pandas as pd
import plotly
import plotly.graph_objs as go
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks


class Sampling:
    def target_info(self, data_frame, feature):
        unique_info = data_frame[feature].value_counts()
        classes = []
        count = []
        for key, value in unique_info.items():
            classes.append(key)
            count.append(value)
        plot_html = self.sample_bar_plot(classes, count)
        return unique_info.to_dict(), plot_html

    def under_sampling(self, data_frame, feature):
        unique_info = data_frame[feature].value_counts()
        classes = []
        count = []
        for key, value in unique_info.items():
            classes.append(key)
            count.append(value)
        min_count = min(count)
        index_of_min = count.index(min_count)
        _class = classes[index_of_min]
        df_test_under = data_frame[data_frame[feature] == _class]
        classes.pop(index_of_min)
        for each in classes:
            data_frame_class = data_frame[data_frame[feature] == each]
            sample_data_frame = data_frame_class.sample(min_count)
            df_test_under = pd.concat([df_test_under, sample_data_frame], axis=0)
        df_test_under.reset_index(inplace=True, drop=True)
        return df_test_under

    def over_sampling(self, data_frame, feature):
        unique_info = data_frame[feature].value_counts()
        classes = []
        count = []
        for key, value in unique_info.items():
            classes.append(key)
            count.append(value)
        max_count = max(count)
        index_of_max = count.index(max_count)
        _class = classes[index_of_max]
        df_test_over = data_frame[data_frame[feature] == _class]
        classes.pop(index_of_max)
        for each in classes:
            data_frame_class = data_frame[data_frame[feature] == each]
            sample_data_frame = data_frame_class.sample(max_count, replace=True)
            df_test_over = pd.concat([df_test_over, sample_data_frame], axis=0)
        df_test_over.reset_index(inplace=True, drop=True)
        return df_test_over

# Return Codes :
# 0 - Response if dataset is perfect
# 1 - Response if Catagorical data is found on the dataset for SMOTE
# 2 - Response if values in a Class are not enough for SMOTE
# 3 - Response if Tomek cant be performed
# 4 - Response if Feature has Float as class values

    def smote(self, data_frame, feature):  # function SMOTE
        try:
            dfcolumns = list(data_frame.select_dtypes(exclude=object))
            loc = dfcolumns.index(feature)  # Storing the Index Position of the Feature

            unique_info = data_frame[feature].value_counts()
            x = data_frame.drop(feature, axis=1)
            y = data_frame[feature]

            dfcolumns.remove(feature)

            smt = SMOTE()
            x_train, y_train = smt.fit_resample(x, y)
            np.bincount(y_train)
        except:
            df = 4
            return df
        else:
            df = pd.DataFrame(np.atleast_2d(x_train), columns=dfcolumns)
            df.insert(loc, feature, y_train)

        return df

    def smote_validate(self, data_frame, feature):  # Validation of labels for each class
        unique_info = data_frame[feature].value_counts()
        flag = 0; # Response if dataset is perfect for SMOTE

        dfc = list(data_frame.select_dtypes(include=object))    # Checking for Categorical Data
        if len(dfc) > 0:
            flag = 1    # Response if Catagorical data is found on the dataset
        else:
            classes = []
            count = []
            for key, value in unique_info.items():
                classes.append(key)
                count.append(value)

            for val in count:   # Checking for samples in each class. Minimum requirement 6
                if val < 6:
                    flag = 2    # Response if values in a Class are not enough for SMOTE
                    break;

        return flag

    def tomek(self, data_frame, feature):   # function Tomek
        try:
            dfcolumns = list(data_frame.select_dtypes(exclude=object))
            loc = dfcolumns.index(feature)  # Storing the Index Position of the Feature

            unique_info = data_frame[feature].value_counts()
            x = data_frame.drop(feature, axis=1)
            y = data_frame[feature]

            dfcolumns.remove(feature)

            tl = TomekLinks()
            x_train, y_train = tl.fit_resample(x, y)
            np.bincount(y_train)

        except:
            return 3    # Response if Tomek cant be performed
        else:
            df = pd.DataFrame(np.atleast_2d(x_train), columns=dfcolumns)
            df.insert(loc, feature, y_train)
            return df   # Response if Tomek is performed

    def sample_bar_plot(self, classes, count):
        data = [go.Bar(x=classes, y=count)]
        plot_html = plotly.offline.plot(data, output_type='div')
        return plot_html
