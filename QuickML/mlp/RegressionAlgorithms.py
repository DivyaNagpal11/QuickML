from .spark_database import *
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics

class Regression:
    def __init__(self, data_frame, project_id):
        self.data_frame = data_frame.dropna()
        self.project_id = project_id

    def target_feature_split(self, target):
        x = self.data_frame.drop(target, axis=1)
        y = self.data_frame[target]
        return x, y

    def data_train_test_split(self, x, y, test_size):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        return x_train, x_test, y_train, y_test

    def feature_drop(self, selected_features):
        for feature in selected_features:
            self.data_frame = self.data_frame.drop(feature, axis=1)

        return self.data_frame

    def filter_features(self, check_features):
        all_features = list(self.data_frame)

        for feature in check_features:
            all_features.remove(feature)

        return all_features


    def data_regression_processing(self, data_options, algorithm_info, selected_features):
        algorithm = algorithm_info[1]
        data_base = SparkDataBase()
        target_feature = data_options[0]

        if selected_features != "None":
            unchecked_features = self.filter_features(selected_features)
            if target_feature in unchecked_features:
                unchecked_features.remove(target_feature)
            frame = self.feature_drop(unchecked_features)

        x, y = self.target_feature_split(target_feature)
        test_size = data_options[1]
        if test_size < 1:
            test_size = float(test_size)
        else:
            test_size = int(test_size)
        x_train, x_test, y_train, y_test = self.data_train_test_split(x, y, test_size)

        df = pd.DataFrame()
        summary, ols_html = self.ols_summary(x, y, df)
        result_data = []
        result_data.append(int(algorithm_info[0]))
        result_data.append(algorithm)
        if algorithm == 'Linear_Regression':
            score = self.linear_regression(x_train, x_test, y_train, y_test, target_feature)
            result_data.append(score)
            return result_data, summary, ols_html, unchecked_features

    def linear_regression(self, x_train, x_test, y_train, y_test, target_feature):
        train_scores = self.ols_train_test_summary(x_train, y_train)
        linreg = LinearRegression()
        linreg = linreg.fit(x_train, y_train)
        y_pred_t = linreg.predict(x_test)

        data_base = SparkDataBase()
        data_base.save_regression_model(self.project_id, "Linear_Regression", round(train_scores[0], 2), round(train_scores[1], 2), target_feature, linreg)
        data = []
        data.append(round(train_scores[0], 2))
        data.append(round(train_scores[1], 2))
        return data

    def ols_summary(self, x, y, df):
        lm = sm.OLS(y, x).fit()
        summary = []
        summary.append(round(lm.rsquared, 3))
        summary.append(round(lm.rsquared_adj, 3))
        summary.append(round(lm.fvalue, 3))
        summary.append(round(lm.aic, 3))
        summary.append(round(lm.bic, 3))

        df['coef'] = round(lm.params, 3)
        df['tvalues'] = round(lm.tvalues, 3)
        df['PValue'] = round(lm.pvalues, 3)

        ols_html = df.to_html(classes=["table", "table-responsive", "table-bordered", "table-hover"])
        return summary, ols_html

    def ols_train_test_summary(self, x, y):
        lm = sm.OLS(y, x).fit()
        summary = []
        summary.append(round(lm.rsquared, 2))
        summary.append(round(lm.rsquared_adj, 2))
        return summary

    def regression_execute_selected_model(self, model, test):
        try:
            predict = model.predict(test)
        except:
            return "Error"
        else:
            return round(predict[0], 2)
