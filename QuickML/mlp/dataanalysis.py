import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np

class ExpDataAnalysis:
    def __init__(self, df):
            self._df = df

    def numerical_summary_from_data_frame(self):
        try:
            a = self._df.describe().transpose()
            a['missing'] = self._df.isnull().sum()
            a['zeros'] = self._df[self._df == 0].count()
            a.round(2)
            return a.to_html(classes=["table", "table-responsive", "table-bordered", "table-hover"])
        except:
            return None

    def categorical_summary_from_data_frame(self):
        try:
            categoricaldataframe = self._df.select_dtypes(include=['object'])
            categoricaldataframe.round(2)
            b = categoricaldataframe.describe().transpose()
            b['missing'] = self._df.isnull().sum()
            return b.to_html(classes=["table", "table-responsive", "table-bordered",  "table-hover"])
        except:
            return None

    def features_list(self):
        return list(self._df)

    def numerical_features(self):
        data = self._df.select_dtypes(exclude=['object'])
        return list(data)

    def categorical_features(self):
        data = self._df.select_dtypes(include=['object'])
        return list(data)

    def correlation_plot(self):
        corr = self._df.corr()
        corr = corr.round(2)
        z = corr.values.tolist()
        x = list(self._df.select_dtypes(exclude=['object']))
        fig = ff.create_annotated_heatmap(z, x=x, y=x, colorscale='Viridis')
        fig['data'][0]['showscale'] = True
        plot_html = plotly.offline.plot(fig, output_type='div')
        return plot_html

    def box_plot(self, x_column, y_column):
        data = [go.Box(y=self._df[y_column], x=self._df[x_column], boxpoints='all', jitter=0.3,
                       pointpos=-1.8)]  # x=self._df[xColumn], name='Boxyy', marker=dict(color='#FF851B') )
        layout = go.Layout(yaxis=dict(title=y_column, zeroline=False),
                           xaxis=dict(title=x_column, zeroline=False), boxmode='group')
        plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
        return plot_html

    def distribution_plot(self, x_column):
        data = [go.Histogram(x=self._df[x_column], histnorm='probability')]
        layout = go.Layout(yaxis=dict(title=x_column, zeroline=False), boxmode='group')
        plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
        return plot_html

    def scatter_plot(self, x_column, y_column):
        data = [go.Scattergl(x=self._df[x_column], y=self._df[y_column], mode='markers',
                             marker=dict(color='#FFBAD2', line=dict(width=1)))]
        layout = go.Layout(go.Layout(yaxis=dict(title=y_column, zeroline=False),
                                     xaxis=dict(title=x_column, zeroline=False), boxmode='group'))
        plot_html = plotly.offline.plot({"data": data, "layout": layout}, output_type='div')
        return plot_html

    def box_plot_outliers(self, feature):
        feature_data = self._df[feature]
        trace0 = go.Box(y=feature_data, name=feature, marker=dict(color='rgb(214, 12, 140)'))
        data = [trace0]
        plot_html = plotly.offline.plot(data, output_type='div')
        return plot_html

    def normal_distribution_plot(self, x_column):
        hist_data = [self._df[x_column]]
        group_labels = [x_column]
        data = ff.create_distplot(hist_data, group_labels, bin_size=0.6, show_rug=False)
        plot_html = plotly.offline.plot({"data": data}, output_type='div')
        return plot_html

    def normal_distribution(self, feature):
        val = np.log(self._df[feature]+1)
        self._df[feature] = np.log(self._df[feature]+1)
        return self._df, val
