from .spark_database import *
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
from sklearn.cluster import KMeans

class Clustering:
    def __init__(self, data_frame, project_id):
        self.data_frame = data_frame.dropna()
        self.project_id = project_id

    def elbow_method(self):
        dfcolumns = list(self.data_frame.select_dtypes(exclude=object))
        df = self.data_frame.values[:, 0:len(dfcolumns)]
        wcss = []

        for i in range(1, 16):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(df)
            wcss.append(kmeans.inertia_)

        X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        trace = go.Scatter(x=X, y=wcss)
        data = [trace]
        elbow_html = plotly.offline.plot(data, output_type='div')
        return elbow_html

    def data_cluster_processing(self, algorithm_info):
        algorithm = algorithm_info[1]
        data_base = SparkDataBase()
        input_parameters = data_base.load_cluster_models_input_parameters(self.project_id, algorithm)
        dfcolumns = list(self.data_frame.select_dtypes(exclude=object))
        df = self.data_frame.values[:, 0:len(dfcolumns)]
        result_data = []
        result_data.append(int(algorithm_info[0]))
        result_data.append(algorithm)
        result_data.append(input_parameters[0])

        if algorithm == 'KMeans':
            label = algorithm_info[2];
            clusters, y_labels = self.KMeans(df, input_parameters, label)
            self.data_frame['Cluster_Labels'] = y_labels
            return result_data, clusters, self.data_frame

    def KMeans(self, df, input_parameters, label):
        dist = input_parameters[4]

        if dist == 'True':
            dist = True
        elif dist == 'False':
            dist = False
        elif dist == 'auto':
            dist = 'auto'

        kmeans = KMeans(n_clusters=input_parameters[0], init=input_parameters[1], n_init=input_parameters[2],
                        max_iter=input_parameters[3], precompute_distances=dist, random_state=input_parameters[5])
        kmeans = kmeans.fit(df)
        y_kmeans = kmeans.labels_
        labels = []

        if label == 'auto':
            for i in range(input_parameters[0]):
                labels.append("Cluster_" + str(i + 1))
        else:
            labels = label.split(',')

        cluster_labels = ",".join(labels)
        plot_html, label_color = self.cluster_scatter_plot(input_parameters[0], df,
                                                           y_kmeans, kmeans.cluster_centers_, labels)

        y_labels = self.cluster_columns(labels, y_kmeans)
        data_base = SparkDataBase()
        data_base.save_cluster_model(self.project_id, 'KMeans', input_parameters[0], cluster_labels,
                                     label_color, kmeans)
        #data_base.update_document(self.data_frame, self.project_id)
        return plot_html, y_labels

    def cluster_scatter_plot(self, k, df, y_kmeans, centroids, labels):

        colors = ['brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'palevioletred',
                  'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'rosybrown', 'teal',
                  'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgrey', 'sienna',
                  'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
                  'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'steelblue',
                  'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink',
                  'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'lightsteelblue',
                  'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'springgreen',
                  'gold', 'goldenrod', 'gray', 'grey', 'green', 'lightgoldenrodyellow', 'wheat',
                  'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'mediumaquamarine',
                  'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'mintcream', 'tan'
                                                                                           'lemonchiffon', 'lightblue',
                  'lightcoral', 'lightcyan', 'mediumslateblue',
                  'lightgray', 'lightgrey', 'lightgreen', 'lightpink', 'lightsalmon', 'violet',
                  'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'turquoise',
                  'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'yellowgreen'
                                                                                    'mediumblue', 'mediumorchid',
                  'mediumpurple', 'mediumseagreen', 'yellow',
                  'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
                  'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'tomato'
                                                                                      'orangered', 'orchid',
                  'palegoldenrod', 'palegreen', 'paleturquoise', 'thistle'
                                                                 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum',
                  'powderblue', 'purple',
                  'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell',
                  'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'whitesmoke']

        clusters = []
        plotdata = []
        color_list = []

        for i in range(k):
            clusters.append(go.Scatter(x=df[y_kmeans == i, 0], y=df[y_kmeans == i, 1],  mode='markers', name=labels[i],
                                       marker=dict(size=8, color=colors[i + k + 3])))
            color_list.append(colors[i + k + 5])
            plotdata.append(clusters[i])

        centers = go.Scatter(x=centroids[:, 0], y=centroids[:,1], mode='markers', name="Centroids",
                             marker=dict(size=12, color='black'))
        plotdata.append(centers)
        data = plotdata
        plot_html = plotly.offline.plot(data, output_type='div')

        label_color = ",".join(color_list)
        return plot_html, label_color

    def cluster_execute_selected_model(self, model, test):
        try:
            predict = model.predict(test)
        except:
            return "Error"
        else:
            label = self.get_pred_cluster_label(predict)
        return label

    def get_pred_cluster_label(self, predict):
        data_base = SparkDataBase()
        labels = data_base.label_cluster(self.project_id)
        retLabels = labels[0]
        label_ret = retLabels.split(',')
        label = label_ret[predict[0]]
        return label

    def cluster_columns(self, labels, y_kmeans):
        y_labels = []
        for i in range(len(y_kmeans)):
            y_labels.append(labels[y_kmeans[i]])

        return y_labels