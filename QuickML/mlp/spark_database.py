import pandas as pd
import json
import pickle
from pathlib import Path
from pyspark.sql import SparkSession,SQLContext
import os
from os import listdir
from os.path import isfile, join
from .unziping import *
from .image_processing import *
import datetime as dt
import collections
import time

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
# sc = spark.sparkContext
sql = SQLContext(spark)
spark.sql('create database IF NOT EXISTS  quickml_database')
spark.sql('use quickml_database')
spark.sql("CREATE TABLE IF NOT EXISTS projects (project_id INT, project_name STRING, project_type STRING, status STRING)")
spark.sql("CREATE TABLE IF NOT EXISTS models_details (project_id INT, algorithm STRING,train_accuracy FLOAT,"
          "test_accuracy FLOAT,accuracy FLOAT ,target_feature STRING)")
spark.sql("CREATE TABLE IF NOT EXISTS models_api (project_id INT, model_id STRING,api_url STRING)")
spark.sql("CREATE TABLE IF NOT EXISTS saved_input_parameters_knn (project_id INT, n_neighbors INT,weights STRING,"
          "algorithm STRING,leaf_size INT,p STRING)")
spark.sql("CREATE TABLE IF NOT EXISTS saved_input_parameters_dt (project_id INT, criterion STRING,splitter STRING,"
          "max_depth INT,min_samples_split INT,min_samples_leaf INT,min_weight_fraction_leaf float,max_features STRING,"
          "random_state INT,max_leaf_nodes INT,min_impurity_decrease FLOAT ,presort STRING)")
spark.sql("CREATE TABLE IF NOT EXISTS saved_input_parameters_rf (project_id INT, n_estimators INT,criterion STRING,"
          "max_depth INT,min_samples_split INT,min_samples_leaf INT,min_weight_fraction_leaf float,max_features STRING,"
          "max_leaf_nodes INT,min_impurity_decrease FLOAT,bootstrap STRING,obb_score STRING,random_state INT,"
          "verbose INT,warm_start STRING)")
spark.sql("CREATE TABLE IF NOT EXISTS saved_input_parameters_svm (project_id INT,C FLOAT,kernel STRING,degree INT,"
          "gamma FLOAT,coef0 FLOAT,shrinking STRING,probability STRING,tol FLOAT,verbose STRING,max_iter INT,"
          "decision_function_shape STRING,random_state INT)")
spark.sql("CREATE TABLE IF NOT EXISTS saved_input_parameters_xgb (project_id INT,max_depth INT,learning_rate FLOAT ,"
          "n_estimators INT,booster STRING,gamma int,min_child_weight int,max_delta_step int,subsample FLOAT,"
          "colsample_bytree FLOAT,colsample_bylevel FLOAT,reg_alpha FLOAT,reg_lambda FLOAT,scale_pos_weight FLOAT ,"
          "base_score FLOAT ,random_state INT)")

#  Table of KMeans Cluster algorithm parameters
spark.sql("CREATE TABLE IF NOT EXISTS saved_input_parameters_cluster_KMeans "
          "(project_id INT, kn_clusters INT, k_init STRING , kn_init INT, "
          "max_iter INT, precompute STRING, random_state INT)")

#  Table of KMeans Cluster algorithm model details
spark.sql("CREATE TABLE IF NOT EXISTS cluster_models_details (project_id INT, algorithm STRING,k_cluster INT,"
          "cluster_labels STRING, label_colors STRING)")

#  Table of KMeans Cluster algorithm model api details
spark.sql("CREATE TABLE IF NOT EXISTS cluster_models_api (project_id INT, model_id STRING,api_url STRING)")

#  Table of Regression model details
spark.sql("CREATE TABLE IF NOT EXISTS regression_models_details (project_id INT, algorithm STRING, train_Rsq FLOAT, "
          "train_adjRsq FLOAT, target_feature STRING)")

#  Table of Regression model api details
spark.sql("CREATE TABLE IF NOT EXISTS regression_models_api (project_id INT, model_id STRING,api_url STRING)")

spark.sql("CREATE TABLE IF NOT EXISTS image_process_model_table(project_id INT,index_no STRING,layer_name STRING, parameters STRING)")
spark.sql("CREATE TABLE IF NOT EXISTS image_process_dense_table(project_id INT,index_no STRING,units STRING, activation STRING)")
spark.sql("CREATE TABLE IF NOT EXISTS image_process_fit_table(project_id INT,l_name STRING, params STRING)")

#  KMeans Cluster algorithm parameters
input_parameters_cluster_KMeans = [['n_clusters', 'input', 'kn_clusters', 8],
                                   ['init', 'select', 'k_init', ['k-means++', 'random']],
                                   ['n_init', 'input', 'kn_init', 10], ['max_iter', 'input', 'kn_init', 300],
                                   ['precompute', 'select', 'k_precompute', ['auto', 'True', 'False']],
                                   ['random_state(0 is None)', 'input', 'rf_random_state', 0]]


input_parameters_knn=[['n_neighbours','input','knn_n_neighbours',5],['weights','select','knn_weights',
                                                                     ['uniform','distance']],
                      ['algorithm','select','knn_algorithm',['auto','ball_tree','kd_tree','brute']],
                      ['leaf_size','input','knn_leaf_size',30],
                      ['p','select','knn_distance_method',['euclidean_distance','manhattan_distance']]]
input_parameters_dt=[
        ['criterion','select','dt_criterion',['gini','entropy']],
        ['splitter','select','dt_splitter',['best','random']],
        ['max_depth(0 is None)','input','dt_max_depth',0],
        ['min_samples_split','input','dt_min_samples_split',2],
        ['min_samples_leaf','input','dt_min_samples_leaf',1],
        ['min_weight_fraction_leaf','input','dt_min_weight_fraction_leaf',0],
        ['max_features','select','dt_max_features',['None','auto','sqrt','log2']],
        ['random_state(0 is None)','input','dt_random_state',0],
        ['max_leaf_nodes(0 is None)','input','dt_max_leaf_nodes',0],
        ['min_impurity_decrease','input','dt_min_impurity_decrease',0],
        ['presort','select','dt_presort',['false','true']]
    ]
input_parameters_rf=[
        ['n_estimators','input','rf_n_estimators',10],
        ['criterion', 'select', 'rf_criterion', ['gini', 'entropy']],
        ['max_depth(0 is None)', 'input', 'rf_max_depth', 0],
        ['min_samples_split', 'input', 'rf_min_samples_split', 2],
        ['min_samples_leaf', 'input', 'rf_min_samples_leaf', 1],
        ['min_weight_fraction_leaf', 'input', 'rf_min_weight_fraction_leaf', 0],
        ['max_features', 'select', 'rf_max_features', ['None', 'auto', 'sqrt', 'log2']],
        ['max_leaf_nodes(0 is None)', 'input', 'rf_max_leaf_nodes', 0],
        ['min_impurity_decrease', 'input', 'rf_min_impurity_decrease', 0],
        ['bootstrap','select','rf_bootstrap',['true','false']],
        ['obb_score','select','rf_obb_score',['false','true']],
        ['random_state(0 is None)', 'input', 'rf_random_state', 0],
        ['verbose','input','rf_verbose',0],
        ['warm_start','select','rf_warm_start',['false','true']]
    ]
input_parameters_svc=[
        ['C','input','svc_c',1.0],
        ['kernel','select','svc_kernel',['rbf','linear','poly','sigmoid']],
        ['degree','input','svc_degree',3],
        ['gamma(0 is auto','input','svc_gamma',0],
        ['coef0','input','svc_coef0',0.0],
        ['shrinking','select','svc_shrinking',['true','false']],
        ['probability','select','svc_probability',['false','true']],
        ['tol','input','svc_tol',0.05],
        ['verbose','select','svc_verbose',['false','true']],
        ['max_iter','input','svc_max_iter',-1],
        ['decision-function_shape','select','svc_decision-function_shape',['ovr','ovo']],
        ['random_state(0 is None)', 'input', 'svc_random_state', 0]
    ]
input_parameters_xgb=[
        ['max_depth','input','xgb_max_depth',3],
        ['learning_rate','input','xgb_learning_rate',0.1],
        ['n_estimators','input','xgb_n_estimators',100],
        ['booster','select','xgb_booster',['gbtree','gblinear','dart']],
        ['gamma','input','xgb_gamma',0],
        ['min_child_weight','input','xgb_min_child_weight',1],
        ['max_delta_step','input','max_delta_step',0],
        ['subsample','input','xgb_subsample',1],
        ['colsample_bytree','input','xgb_colsample_bytree',1],
        ['colsample_bylevel','input','xgb_colsample_bylevel',1],
        ['reg_alpha','input','xgb_reg_alpha',0],
        ['reg_lambda','input','xgb_lambda',1],
        ['scale_pos_weight','input','xgb_scale_pos_weight',1],
        ['base_score','input','xgb_base_score',0.5],
        ['random_state','input','xgb_random_state',0]
    ]


class SparkDataBase:
    def projects_info(self):
        projects_df = spark.sql("select * from projects where status='active'")
        project_id_row = projects_df.select("project_id").collect()
        project_names_row = projects_df.select("project_name").collect()
        project_ids = []
        project_names = []
        for p_id, p_name in zip(project_id_row, project_names_row):
            project_ids.append(p_id.project_id)
            project_names.append(p_name.project_name)
        return project_ids, project_names

    def retreive_project_name(self, project_id):
        status = "active"
        project_name = spark.sql("select project_name from projects where project_id = {0} and status='{1}'".format(project_id, status)).first()[0]
        return project_name

    def retreive_project_type(self, project_id):
        status = "active"
        project_type = spark.sql("select project_type from projects where project_id = {0} and status='{1}'".format(project_id, status)).first()[0]
        return project_type

    def create_id(self):
        a = spark.sql("select count(*) from projects").first()[0]
        if a == 0:
            project_id = 1
        else:
            project_id = a+1
        return project_id

    def read_data(self, name, file, project_id, status,type):
        filename, extension = str(file).split('.')
        if type == "Image Processing":
            spark.sql(
                "INSERT INTO projects VALUES ({0},'{1}','{2}', '{3}')".format(project_id, name, type,
                                                                              status))
            unzip(file, project_id)
            create_img_labels(project_id)

        elif type == "Data Processing":
            try:
                if extension == "xlsx" or extension == "xls":
                    data_frame = pd.read_excel(file, na_values=['?'])
                else:
                    data_frame = pd.read_csv(file, na_values=['?'])
                columnns = list(data_frame)
                for i in range(0, len(columnns)):
                    columnns[i] = str.strip(columnns[i])
                    columnns[i] = columnns[i].replace(' ', '_')
                data_frame.columns = columnns
                spark_df = sql.createDataFrame(data_frame)
                spark_df.write.mode('overwrite').saveAsTable('org_df' + str(project_id))
                spark_df.write.mode('overwrite').saveAsTable('cur_df' + str(project_id))
                spark_df.write.mode('overwrite').saveAsTable('osb_df' + str(project_id))
            except Exception as e:
                raise Exception()
            else:
                spark.sql("INSERT INTO projects VALUES ({0},'{1}', '{2}','{3}')".format(project_id, name, "Data Processing", status))

        else:
            try:
                if extension == "xlsx" or extension == "xls":
                    ah_data = pd.read_excel(file, na_values=['?'])

                else:
                    ah_data = pd.read_csv(file, na_values=['?'])

                ah_data['posted_date'] = pd.to_datetime(ah_data['posted_date'])
                ah_data = ah_data.sort_values('posted_date').reset_index(drop=True)

                columns = list(ah_data)
                for i in range(0, len(columns)):
                    columns[i] = str.strip(columns[i])
                    columns[i] = columns[i].replace(' ', '_')

                ah_data.columns = columns


                spark_df = sql.createDataFrame(ah_data)


                spark_df.write.mode('overwrite').saveAsTable('org_df' + str(project_id))
                spark_df.write.mode('overwrite').saveAsTable('cur_df' + str(project_id))
                spark_df.write.mode('overwrite').saveAsTable('osb_df' + str(project_id))
            except Exception as e:
                raise Exception()
            else:
                spark.sql(
                    "INSERT INTO projects VALUES ({0},'{1}', '{2}','{3}')".format(project_id, name, "Cash Forecasting",
                                                                                  status))

    def update_document(self, data_frame, project_id):
        one_step_back_data_frame = self.find_data_frame(project_id)
        one_step_back_data_frame = sql.createDataFrame(one_step_back_data_frame)
        self.update_one_step_back_data_frame(project_id, one_step_back_data_frame)
        data_frame = sql.createDataFrame(data_frame)
        data_frame.write.mode('overwrite').saveAsTable('cur_df' + str(project_id))

    def find_data_frame(self, project_id):
        spark_dataframe = sql.table('cur_df' + str(project_id))
        data_frame = spark_dataframe.toPandas()
        return data_frame

    def update_current_data_frame_with_one_step_back(self, project_id):
        data_frame = sql.table('osb_df' + str(project_id))
        data_frame.write.mode('overwrite').saveAsTable('cur_df' + str(project_id))

    def update_one_step_back_data_frame(self, project_id, data_frame):
        data_frame.write.mode("overwrite").saveAsTable('osb_df' + str(project_id))

    def save_model(self, project_id, algorithm,train_accuracy,test_accuracy,accuracy,target_feature,model):
        a = spark.sql("select count(*) from models_details where project_id={0} and algorithm='{1}'".format(project_id,algorithm)).first()[0]
        if a != 0:
            spark.sql("INSERT OVERWRITE TABLE models_details SELECT * FROM models_details WHERE project_id!={0} or algorithm!='{1}'".format(project_id,algorithm))
        spark.sql("INSERT INTO models_details VALUES({0},'{1}',{2},{3},{4},'{5}')".format(project_id, algorithm, train_accuracy, test_accuracy, accuracy,target_feature))
        file_path="saved_models/"
        model_name = ""
        for each in algorithm.split(" "):
            model_name += each[0]
        model_name += str(project_id)+'.pkl'
        file_path += model_name
        with open(Path(file_path),'wb') as handle:
            pickle.dump(model, handle)

    def load_model(self, project_id, algorithm):
        file_path = r"saved_models/"
        model_name = ""
        for each in algorithm.split(" "):
            model_name += each[0]
        model_name += str(project_id) + '.pkl'
        file_path += model_name
        with open(Path(file_path), 'rb') as handle:
            model = pickle.load(handle)
        return model

    def load_models_accuracy(self,project_id):
        a = spark.sql("select count(*) from models_details where project_id={0}".format(project_id)).first()[0]
        if a == 0:
            return 0
        else:
            res_df = spark.sql("SELECT * FROM models_details where project_id={0}".format(project_id))
            res_df = res_df.drop('project_id')
            algorithm = []
            train_accuracy = []
            test_accuracy = []
            accuracy = []
            for each in res_df.select('algorithm').collect():
                algorithm.append(each.algorithm)
            for each in res_df.select('train_accuracy').collect():
                train_accuracy.append(each.train_accuracy)
            for each in res_df.select('test_accuracy').collect():
                test_accuracy.append(each.test_accuracy)
            for each in res_df.select('accuracy').collect():
                accuracy.append(each.accuracy)
            res_data = []
            for i in range(0,len(accuracy)):
                new = {}
                new['algorithm'] = algorithm[i]
                new['train_accuracy'] = round(train_accuracy[i],2)
                new['test_accuracy'] = round(test_accuracy[i],2)
                new['accuracy'] = round(accuracy[i],2)
                res_data.append(new)
            return res_data

    def default_input_parameters(self, algorithm):
        algorithms = ['K Nearest Neighbour', 'Decision Tree', 'Random Forest', 'Support Vector Machine', 'XgBoost']
        if algorithm == algorithms[0]:
            return input_parameters_knn
        elif algorithm == algorithms[1]:
            return input_parameters_dt
        elif algorithm == algorithms[2]:
            return input_parameters_rf
        elif algorithm == algorithms[3]:
            return input_parameters_svc
        else:
            return input_parameters_xgb

    def load_models_input_parameters(self, project_id, algorithm):
        algorithms = ['K Nearest Neighbour', 'Decision Tree', 'Random Forest', 'Support Vector Machine', 'XgBoost']
        if algorithm == algorithms[0]:
            res_df = spark.sql("select * from saved_input_parameters_knn where project_id={0}".format(project_id))
            res_df = res_df.drop('project_id')
            input_parameters = []
            for i in range(0,len(res_df.columns)):
                input_parameters.append(res_df.first()[i])
            return input_parameters
        elif algorithm == algorithms[1]:
            res_df = spark.sql("select * from saved_input_parameters_dt where project_id={0}".format(project_id))
            res_df = res_df.drop('project_id')
            input_parameters = []
            for i in range(0, len(res_df.columns)):
                input_parameters.append(res_df.first()[i])
            return input_parameters
        elif algorithm == algorithms[2]:
            res_df = spark.sql("select * from saved_input_parameters_rf where project_id={0}".format(project_id))
            res_df = res_df.drop('project_id')
            input_parameters = []
            for i in range(0, len(res_df.columns)):
                input_parameters.append(res_df.first()[i])
            return input_parameters
        elif algorithm == algorithms[3]:
            res_df = spark.sql("select * from saved_input_parameters_svm where project_id={0}".format(project_id))
            res_df = res_df.drop('project_id')
            input_parameters = []
            for i in range(0, len(res_df.columns)):
                input_parameters.append(res_df.first()[i])
            return input_parameters
        else:
            res_df = spark.sql("select * from saved_input_parameters_xgb where project_id={0}".format(project_id))
            res_df = res_df.drop('project_id')
            input_parameters = []
            for i in range(0, len(res_df.columns)):
                input_parameters.append(res_df.first()[i])
            return input_parameters

    def save_input_parameters(self, project_id, algorithm, input_parameters):
        algorithms = ['K Nearest Neighbour', 'Decision Tree', 'Random Forest', 'Support Vector Machine', 'XgBoost']
        if algorithm == algorithms[0]:
            a = spark.sql(
                "select count(*) from saved_input_parameters_knn where project_id={0}".format(project_id)).first()[0]
            if a != 0:
                spark.sql(
                    "INSERT OVERWRITE TABLE saved_input_parameters_knn SELECT * FROM saved_input_parameters_knn WHERE not project_id={0}".format(project_id))
            spark.sql("INSERT INTO saved_input_parameters_knn VALUES({0},{1},'{2}','{3}',{4},'{5}')".
                      format(project_id, input_parameters[0], input_parameters[1], input_parameters[2], input_parameters[3],
                             input_parameters[4]))
        elif algorithm == algorithms[1]:
            a = spark.sql(
                "select count(*) from saved_input_parameters_dt where project_id={0}".format(project_id)).first()[0]
            if a != 0:
                spark.sql(
                    "INSERT OVERWRITE TABLE saved_input_parameters_dt SELECT * FROM saved_input_parameters_dt WHERE not project_id={0}".format(
                        project_id))
            spark.sql("INSERT INTO saved_input_parameters_dt VALUES({0},'{1}','{2}',{3},{4},{5},{6},'{7}',{8},{9},{10},'{11}')".
                      format(project_id, input_parameters[0], input_parameters[1], input_parameters[2],
                             input_parameters[3], input_parameters[4],input_parameters[5],input_parameters[6],
                             input_parameters[7],input_parameters[8],input_parameters[9],input_parameters[10]))
        elif algorithm == algorithms[2]:
            a = spark.sql(
                "select count(*) from saved_input_parameters_rf where project_id={0}".format(project_id)).first()[0]
            if a != 0:
                spark.sql(
                    "INSERT OVERWRITE TABLE saved_input_parameters_rf SELECT * FROM saved_input_parameters_rf WHERE not project_id={0}".format(
                        project_id))
            spark.sql(
                "INSERT INTO saved_input_parameters_rf VALUES({0},{1},'{2}',{3},{4},{5},{6},'{7}',{8},{9},'{10}','{11}',{12},{13},'{14}')".
                format(project_id, input_parameters[0], input_parameters[1], input_parameters[2],input_parameters[3],
                       input_parameters[4], input_parameters[5], input_parameters[6],input_parameters[7],
                       input_parameters[8], input_parameters[9], input_parameters[10],input_parameters[11],
                       input_parameters[12],input_parameters[13]))
        elif algorithm == algorithms[3]:
            a = spark.sql(
                "select count(*) from saved_input_parameters_svm where project_id={0}".format(project_id)).first()[0]
            if a != 0:
                spark.sql(
                    "INSERT OVERWRITE TABLE saved_input_parameters_svm SELECT * FROM saved_input_parameters_svm WHERE not project_id={0}".format(
                        project_id))
            spark.sql(
                "INSERT INTO saved_input_parameters_svm VALUES({0},{1},'{2}',{3},{4},{5},'{6}','{7}',{8},'{9}',{10},'{11}',{12})".
                format(project_id, input_parameters[0], input_parameters[1], input_parameters[2],
                       input_parameters[3],input_parameters[4], input_parameters[5], input_parameters[6],
                       input_parameters[7], input_parameters[8], input_parameters[9], input_parameters[10],
                       input_parameters[11]))
        else:
            a = spark.sql(
                "select count(*) from saved_input_parameters_xgb where project_id={0}".format(project_id)).first()[0]
            if a != 0:
                spark.sql(
                    "INSERT OVERWRITE TABLE saved_input_parameters_xgb SELECT * FROM saved_input_parameters_xgb WHERE not project_id={0}".format(
                        project_id))
            spark.sql(
                "INSERT INTO saved_input_parameters_xgb VALUES({0},{1},{2},{3},'{4}',{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15})".
                format(project_id, input_parameters[0], input_parameters[1], input_parameters[2],
                       input_parameters[3],input_parameters[4], input_parameters[5], input_parameters[6],
                       input_parameters[7],input_parameters[8], input_parameters[9], input_parameters[10],
                       input_parameters[11],input_parameters[12], input_parameters[13], input_parameters[14]))

    def save_model_for_api(self,project_id,model_id,api_url):
        a = spark.sql("select count(*) from models_api where model_id='{0}'".format(model_id)).first()[0]
        if a != 0:
            spark.sql(
                "INSERT OVERWRITE TABLE models_api SELECT * FROM models_api WHERE model_id!='{0}'".format(model_id))
        spark.sql("INSERT INTO models_api VALUES({0},'{1}','{2}')".format(project_id, model_id, api_url))

    def retrive_models_api_url(self,model_id):
        a = spark.sql("select count(*) from models_api where model_id='{0}'".format(model_id)).first()[0]
        if a == 0:
            return 0
        else:
            res_df=spark.sql("select api_url from models_api where model_id='{0}'".format(model_id))
            return res_df.first()[0]

    def retrive_saved_model_api(self,model_id):
        file_path = "saved_models/"
        file_path += model_id+'.pkl'
        with open(Path(file_path), 'rb') as handle:
            model = pickle.load(handle)
        return model

    def target_feature(self,project_id):
        target_feature = spark.sql("SELECT target_feature FROM models_details where project_id={0}".format(project_id)).first()[0]
        return target_feature

    def saved_model_for_project(self,project_id):
        res_df = spark.sql("select algorithm from models_details where project_id={0}".format(project_id)).select('algorithm').collect()
        saved_algorithms=[]
        for each in res_df:
            saved_algorithms.append(each.algorithm)
        return saved_algorithms

    def cluster_input_parameters(self, algorithm):
        algorithms = ['KMeans']
        if algorithm == algorithms[0]:
            return input_parameters_cluster_KMeans

    def save_cluster_input_parameters(self, project_id, algorithm, input_parameters):
        algorithms = ['KMeans']
        if algorithm == algorithms[0]:
            a = spark.sql(
                "select count(*) from saved_input_parameters_cluster_KMeans where project_id={0}".format(project_id)).first()[0]
            if a != 0:
                spark.sql(
                    "INSERT OVERWRITE TABLE saved_input_parameters_cluster_KMeans SELECT * FROM saved_input_parameters_cluster_KMeans WHERE not project_id={0}".format(project_id))
            spark.sql("INSERT INTO saved_input_parameters_cluster_KMeans VALUES({0},{1},'{2}',{3},{4},'{5}',{6})".
                      format(project_id, input_parameters[0], input_parameters[1], input_parameters[2],
                             input_parameters[3], input_parameters[4], input_parameters[5]))

    def load_cluster_models_input_parameters(self, project_id, algorithm):
        algorithms = ['KMeans']
        if algorithm == algorithms[0]:
            res_df = spark.sql("select * from saved_input_parameters_cluster_KMeans where project_id={0}".format(project_id))
            res_df = res_df.drop('project_id')
            input_parameters = []
            for i in range(0, len(res_df.columns)):
                input_parameters.append(res_df.first()[i])
            return input_parameters

    def save_cluster_model(self, project_id, algorithm, k_cluster, cluster_labels, label_colors, model):
        a = spark.sql("select count(*) from cluster_models_details where project_id={0} and algorithm='{1}'"
                      .format(project_id,algorithm)).first()[0]
        if a != 0:
            spark.sql(
                "INSERT OVERWRITE TABLE cluster_models_details SELECT * FROM cluster_models_details WHERE project_id!={0} or algorithm!='{1}'"
                    .format(project_id, algorithm))
        spark.sql("INSERT INTO cluster_models_details VALUES({0},'{1}',{2},'{3}','{4}')"
                  .format(project_id, algorithm, k_cluster, cluster_labels, label_colors))

        file_path = "saved_models/"
        model_name = ""
        for each in algorithm.split(" "):
            model_name += each
        model_name += str(project_id) + '.pkl'
        file_path += model_name
        with open(Path(file_path), 'wb') as handle:
            pickle.dump(model, handle)

    def load_cluster_model(self, project_id, algorithm):
        file_path = r"saved_models/"
        model_name = ""
        for each in algorithm.split(" "):
            model_name += each
        model_name += str(project_id) + '.pkl'
        file_path += model_name
        with open(Path(file_path), 'rb') as handle:
            model = pickle.load(handle)
        return model

    def load_models_cluster(self,project_id):
        a = spark.sql("select count(*) from cluster_models_details where project_id={0}".format(project_id)).first()[0]
        if a == 0:
            return 0
        else:
            res_df = spark.sql("SELECT * FROM cluster_models_details where project_id={0}".format(project_id))
            res_df = res_df.drop('project_id')
            algorithm = []
            k_clusters = []
            cluster_labels = []
            label_colors = []
            for each in res_df.select('algorithm').collect():
                algorithm.append(each.algorithm)
            for each in res_df.select('k_cluster').collect():
                k_clusters.append(each.k_cluster)
            for each in res_df.select('cluster_labels').collect():
                cluster_labels.append(each.cluster_labels)
            for each in res_df.select('label_colors').collect():
                label_colors.append(each.label_colors)

            res_data=[]

            for i in range(0,len(algorithm)):
                new={}
                new['algorithm']=algorithm[i]
                new['k_cluster'] = k_clusters[i]
                new['cluster_labels'] = cluster_labels[i]
                new['label_colors'] = label_colors[i]
                res_data.append(new)
            return res_data

    def saved_cluster_model_for_project(self,project_id):
        res_df = spark.sql("select algorithm from cluster_models_details where project_id={0}"
                           .format(project_id)).select('algorithm').collect()
        saved_algorithms = []
        for each in res_df:
            saved_algorithms.append(each.algorithm)
        return saved_algorithms

    def label_cluster(self,project_id):
        label_ret = spark.sql("select cluster_labels from cluster_models_details where project_id={0}"
                              .format(project_id)).select('cluster_labels').collect()
        labels = []
        for each in label_ret:
            labels.append(each.cluster_labels)
        return labels

    def cluster_save_model_for_api(self, project_id, model_id, api_url):
        a = spark.sql("select count(*) from cluster_models_api where model_id='{0}'".format(model_id)).first()[0]
        if a != 0:
            spark.sql(
                "INSERT OVERWRITE TABLE cluster_models_api SELECT * FROM cluster_models_api WHERE model_id!='{0}'".format(model_id))
        spark.sql("INSERT INTO cluster_models_api VALUES({0},'{1}','{2}')".format(project_id, model_id, api_url))

    def cluster_retrive_models_api_url(self, model_id):
        a = spark.sql("select count(*) from cluster_models_api where model_id='{0}'".format(model_id)).first()[0]
        if a == 0:
            return 0
        else:
            res_df=spark.sql("select api_url from cluster_models_api where model_id='{0}'".format(model_id))
            return res_df.first()[0]

    def cluseter_retrive_saved_model_api(self, model_id):
        file_path = "saved_models/"
        file_path += model_id+'.pkl'
        with open(Path(file_path), 'rb') as handle:
            model = pickle.load(handle)
        return model

    def testModelApi(self): #   Function to check values stored in cluster_models_details and cluster_models_api tables
        res_df1 = spark.sql("Select * from cluster_models_api ")
        projectid1 = []
        modelId = []
        apiurl = []
        for each in res_df1.select('project_id').collect():
            projectid1.append(each.project_id)
        for each in res_df1.select('model_id').collect():
            modelId.append(each.model_id)
        for each in res_df1.select('api_url').collect():
            apiurl.append(each.api_url)

        res_data1 = []

        for i in range(0, len(projectid1)):
            new = {}
            new['projectid'] = projectid1[i]
            new['model_id'] = modelId[i]
            new['api_url'] = apiurl[i]
            res_data1.append(new)

        res_df = spark.sql("SELECT * FROM cluster_models_details")

        projectid = []
        algorithm = []
        k_clusters = []
        cluster_labels = []
        label_colors = []
        for each in res_df.select('project_id').collect():
            projectid.append(each.project_id)
        for each in res_df.select('algorithm').collect():
            algorithm.append(each.algorithm)
        for each in res_df.select('k_cluster').collect():
            k_clusters.append(each.k_cluster)
        for each in res_df.select('cluster_labels').collect():
            cluster_labels.append(each.cluster_labels)
        for each in res_df.select('label_colors').collect():
            label_colors.append(each.label_colors)

        res_data = []

        for i in range(0, len(algorithm)):
            new = {}
            new['projectid'] = projectid[i]
            new['algorithm'] = algorithm[i]
            new['k_cluster'] = k_clusters[i]
            new['cluster_labels'] = cluster_labels[i]
            new['label_colors'] = label_colors[i]
            res_data.append(new)

    def save_regression_model(self, project_id, algorithm, train_Rsq, train_adjRsq, target_feature, model):
        a = spark.sql("select count(*) from regression_models_details where project_id={0} and algorithm='{1}'"
                      .format(project_id, algorithm)).first()[0]
        if a != 0:
            spark.sql(
                "INSERT OVERWRITE TABLE regression_models_details SELECT * FROM regression_models_details WHERE project_id!={0} or algorithm!='{1}'"
                    .format(project_id, algorithm))
        spark.sql("INSERT INTO regression_models_details VALUES({0},'{1}',{2},{3},'{4}')"
                  .format(project_id, algorithm, train_Rsq, train_adjRsq, target_feature))

        file_path = "saved_models/"
        model_name = ""
        for each in algorithm.split(" "):
            model_name += each
        model_name += str(project_id) + '.pkl'
        file_path += model_name
        with open(Path(file_path), 'wb') as handle:
            pickle.dump(model, handle)

    def load_regression_model_api(self, project_id, algorithm):
        file_path = r"saved_models/"
        model_name = ""
        for each in algorithm.split(" "):
            model_name += each
        model_name += str(project_id) + '.pkl'
        file_path += model_name
        with open(Path(file_path), 'rb') as handle:
            model = pickle.load(handle)
        return model

    def load_models_regression_views(self, project_id):
        a = spark.sql("select count(*) from regression_models_details where project_id={0}".format(project_id)).first()[0]
        if a == 0:
            return 0
        else:
            res_df = spark.sql("SELECT * FROM regression_models_details where project_id={0}".format(project_id))
            res_df = res_df.drop('project_id')
            algorithm = []
            train_Rsqs = []
            train_adjRsqs = []
            for each in res_df.select('algorithm').collect():
                algorithm.append(each.algorithm)
            for each in res_df.select('train_Rsq').collect():
                train_Rsqs.append(each.train_Rsq)
            for each in res_df.select('train_adjRsq').collect():
                train_adjRsqs.append(each.train_adjRsq)

            res_data = []

            for i in range(0,len(algorithm)):
                new = {}
                new['algorithm'] = algorithm[i]
                new['train_Rsq'] = round(train_Rsqs[i],2)
                new['train_adjRsq'] = round(train_adjRsqs[i],2)
                res_data.append(new)
            return res_data

    def regression_save_model_for_api(self, project_id, model_id, api_url):
        a = spark.sql("select count(*) from regression_models_api where model_id='{0}'".format(model_id)).first()[0]
        if a != 0:
            spark.sql(
                "INSERT OVERWRITE TABLE regression_models_api SELECT * FROM regression_models_api WHERE model_id!='{0}'".format(model_id))
        spark.sql("INSERT INTO regression_models_api VALUES({0},'{1}','{2}')".format(project_id, model_id, api_url))

    def regression_retrive_models_api_url(self, model_id):
        a = spark.sql("select count(*) from regression_models_api where model_id='{0}'".format(model_id)).first()[0]
        if a == 0:
            return 0
        else:
            res_df=spark.sql("select api_url from regression_models_api where model_id='{0}'".format(model_id))
            return res_df.first()[0]

    def regression_target_feature(self,project_id):
        target_feature = spark.sql("SELECT target_feature FROM regression_models_details where project_id={0}".format(project_id)).first()[0]
        return target_feature

    def retrieve_saved_regression_model_for_project(self, project_id):
        res_df = spark.sql("select algorithm from regression_models_details where project_id={0}"
                           .format(project_id)).select('algorithm').collect()
        saved_algorithms = []
        for each in res_df:
            saved_algorithms.append(each.algorithm)
        return saved_algorithms

    def Delete_project(self, project_id):
        project_type = self.retreive_project_type(project_id)
        if project_type == 'Data Processing':
            reg = spark.sql("select count(*) from regression_models_details where project_id={0}".format(project_id)).first()[0]
            if reg != 0:
                spark.sql("INSERT OVERWRITE TABLE regression_models_details SELECT * FROM regression_models_details "
                          "WHERE project_id!={0}".format(project_id))
                spark.sql("INSERT OVERWRITE TABLE regression_models_api SELECT * FROM regression_models_api WHERE "
                          "project_id!='{0}'".format(project_id))

            clus = spark.sql("select count(*) from cluster_models_details where project_id={0}"
                             .format(project_id)).first()[0]
            if clus != 0:
                spark.sql("INSERT OVERWRITE TABLE cluster_models_details SELECT * FROM cluster_models_details "
                          "WHERE project_id!={0}".format(project_id))
                spark.sql("INSERT OVERWRITE TABLE saved_input_parameters_cluster_KMeans SELECT * "
                          "FROM saved_input_parameters_cluster_KMeans WHERE not project_id={0}".format(project_id))
                spark.sql("INSERT OVERWRITE TABLE cluster_models_api SELECT * FROM cluster_models_api "
                          "WHERE project_id!='{0}'".format(project_id))

            clas = spark.sql("select count(*) from models_details where project_id={0}".format(project_id)).first()[0]
            if clas != 0:
                spark.sql("INSERT OVERWRITE TABLE models_details SELECT * FROM models_details "
                          "WHERE project_id!={0}".format(project_id))
                spark.sql("INSERT OVERWRITE TABLE models_api SELECT * FROM models_api "
                          "WHERE project_id!='{0}'".format(project_id))
                spark.sql("INSERT OVERWRITE TABLE saved_input_parameters_knn SELECT * FROM saved_input_parameters_knn "
                          "WHERE not project_id={0}".format(project_id))
                spark.sql("INSERT OVERWRITE TABLE saved_input_parameters_dt SELECT * FROM saved_input_parameters_dt "
                          "WHERE not project_id={0}".format(project_id))
                spark.sql("INSERT OVERWRITE TABLE saved_input_parameters_rf SELECT * FROM saved_input_parameters_rf "
                          "WHERE not project_id={0}".format(project_id))
                spark.sql("INSERT OVERWRITE TABLE saved_input_parameters_svm SELECT * FROM saved_input_parameters_svm "
                          "WHERE not project_id={0}".format(project_id))
                spark.sql("INSERT OVERWRITE TABLE saved_input_parameters_xgb SELECT * FROM saved_input_parameters_xgb "
                          "WHERE not project_id={0}".format(project_id))

            mypath = r"saved_models/"
            modelId = self.get_saved_models(project_id, mypath)

            try:
                for model in modelId:
                    os.remove(r"saved_models/" + model)
            except:
                Msg = "Error"
            else:
                Msg = "Deleted"



        project_name = self.retreive_project_name(project_id)
        status = "inactive"
        spark.sql("INSERT OVERWRITE TABLE projects SELECT * FROM projects "
                  "WHERE not project_id={0}".format(project_id))
        spark.sql("INSERT INTO projects VALUES ({0},'{1}','{2}', '{3}')".format(project_id, project_name, project_type, status))

        return Msg

    def get_saved_models(self, project_id, mypath):
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        delFile = []

        for each in onlyfiles:
            pid = ""
            for i in each:
                try:
                    le = int(i)
                except:
                    continue
                else:
                    pid += str(le)
                if int(pid) == project_id:
                    delFile.append(each)
                    
        return delFile

    def create_index_model(self, project_id):
        a = spark.sql("select max(index_no+0) from image_process_model_table where project_id={0}".format(project_id)).first()[0]
        if a is None:
            index = 1
        else:
            index = int(a)+1
        return index

    def image_process_model_func(self, project_id, index_no, layer_name, parameters):
        spark.sql("INSERT INTO image_process_model_table VALUES ({0},'{1}','{2}','{3}')".format(project_id, index_no, layer_name, parameters))

    def create_index_dense(self, project_id):
        a = spark.sql("select max(index_no+0) from image_process_dense_table where project_id={0}".format(project_id)).first()[0]
        if a is None:
            index = 1
        else:
            index = int(a)+1
        return index

    def image_process_dense_func(self, project_id, index_no, units, activation):
        spark.sql("INSERT INTO image_process_dense_table VALUES ({0},'{1}','{2}','{3}')".format(project_id, index_no,
                                                                                                units, activation))

    def image_process_fit_func(self, project_id, l_name, params):
        a = spark.sql("select count(*) from image_process_fit_table where project_id={0} and l_name='{1}'"
                      .format(project_id, l_name)).first()[0]
        if a != 0:
            c = spark.sql("INSERT OVERWRITE TABLE image_process_fit_table SELECT * FROM image_process_fit_table "
                          "WHERE project_id!={0} or l_name!='{1}'".format(project_id, l_name))
        d = spark.sql("INSERT INTO image_process_fit_table VALUES({0},'{1}','{2}')".format(project_id, l_name, params))

    def show_list(self, project_id, layer_name):
        df = spark.sql("select * from image_process_model_table where project_id={0} and layer_name='{1}'order by index_no+0 ASC"
                       .format(project_id, layer_name))
        return df

    def dense_table_show_list(self, project_id):
        df = spark.sql("select * from image_process_dense_table where project_id={0} order by index_no+0 ASC"
                       .format(project_id))
        return df

    def fit_table_show_list(self, project_id, layer_name):
        df = spark.sql("select * from image_process_fit_table where project_id={0} and l_name='{1}'"
                       .format(project_id, layer_name))
        return df

