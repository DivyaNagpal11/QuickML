from django.shortcuts import render
from django.http import JsonResponse,HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pickle
import pandas as pd
from mlp.spark_database import *
from mlp.ClusterAlgorithms import *
from mlp.RegressionAlgorithms import *

@csrf_exempt
def predict_api(request, model_id):
    received_json_data = json.loads(request.body.decode("utf-8"))
    predict_data = pd.DataFrame(received_json_data, index=[0])
    data_base = SparkDataBase()
    data = data_base.retrive_models_api_url(model_id)
    if data == 0:
        return HttpResponse("No Model is present with that id")
    else:
        model = data_base.retrive_saved_model_api(model_id)
        predicted = model.predict(predict_data)
        return HttpResponse(predicted)

@csrf_exempt
def cluster_predict_api(request, model_id):
    received_json_data = json.loads(request.body.decode("utf-8"))
    predict_data = pd.DataFrame(received_json_data, index=[0])
    data_base = SparkDataBase()
    project_id = get_projectId(model_id)
    if project_id == "":
        return HttpResponse("No project is present with that id")
    else:
        data_frame = data_base.find_data_frame(project_id)
        execute_algorithms = Clustering(data_frame, project_id)
        data = data_base.cluster_retrive_models_api_url(model_id)
        if data == 0:
            return HttpResponse("No Model is present with that id")
        else:
            model = data_base.cluseter_retrive_saved_model_api(model_id)
            predicted = execute_algorithms.cluster_execute_selected_model(model, predict_data)
            if predicted == 'Error':
                return HttpResponse("Feature count mismatch between Model Trained and Features Passed")
            else:
                return HttpResponse(predicted)

@csrf_exempt
def get_projectId(model_id):
    project_id = ""
    for each in model_id:
        try:
            le = int(each)
        except:
            continue
        else:
            project_id += str(le)

    return project_id

@csrf_exempt
def regression_predict_api(request, model_id):
    received_json_data = json.loads(request.body.decode("utf-8"))
    predict_data = pd.DataFrame(received_json_data, index=[0])
    data_base = SparkDataBase()
    project_id, algorithm = get_project_Algorithm(model_id)
    if project_id == "" or algorithm == "":
        return HttpResponse("No project is present with that id")
    else:
        data_frame = data_base.find_data_frame(project_id)
        regression = Regression(data_frame, project_id)
        data = data_base.regression_retrive_models_api_url(model_id)
        if data == 0:
            return HttpResponse("No Model is present with that id")
        else:
            model = data_base.load_regression_model_api(project_id, algorithm)
            predicted = regression.regression_execute_selected_model(model, predict_data)

            if predicted == 'Error':
                return HttpResponse("Feature count mismatch between Model Trained and Features Passed")
            else:
                return HttpResponse(predicted)

@csrf_exempt
def get_project_Algorithm(model_id):
    project_id = ""
    algorithm = ""
    for each in model_id:
        try:
            le = int(each)
        except:
            algorithm += each
            continue
        else:
            project_id += str(le)

    return project_id, algorithm
