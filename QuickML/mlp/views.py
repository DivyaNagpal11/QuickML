from django.http import JsonResponse,HttpResponse
from django.http import StreamingHttpResponse
from django.shortcuts import render, redirect
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt
from .dataanalysis import *
from .normalization import *
from .missingvalues import *
from .encoding import *
from .sampling import *
from .algorithms import *
from .spark_database import *
from .binning import *
from .ClusterAlgorithms import *
from .RegressionAlgorithms import *
import csv
import threading
from .image_processing import *
import os
from django.core.files import File
from PIL import Image, ImageOps
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from keras.models import load_model
from .cash_functions import *
from dateutil.relativedelta import relativedelta


@csrf_exempt
def main(request):
    return render(request, 'project.html')


@csrf_exempt
def project_names(request):
    data_base = SparkDataBase()
    project_ids, project_name = data_base.projects_info()
    projects = zip(project_ids, project_name)
    context = {'projects': projects}
    drop_down_html = render_to_string('Controls/dropdownlist.html', context, request=request)
    return JsonResponse({'drop_down_html': drop_down_html})


@csrf_exempt
def project(request, value):
    project_id = int(value)
    data_base = SparkDataBase()
    project_name = data_base.retreive_project_name(project_id)
    context = {'p_name': project_name, 'id': project_id}
    return render(request, 'project.html', context=context)


@csrf_exempt
def create_project(request):
    name = request.POST.get('project_name')
    type = request.POST.get('project_type')
    file = request.FILES["project_file"]
    data_base = SparkDataBase()
    project_id = data_base.create_id()
    status = 'active'
    data_base.read_data(name, file, project_id, status,type)
    redirect_url = '/project/' + str(project_id)
    return JsonResponse({"new_url":redirect_url})



@csrf_exempt
def getSidebar(request):
    project_id = request.POST['project_id']
    data_base = SparkDataBase()
    project_type = data_base.retreive_project_type(int(project_id))
    if project_type == "Data Processing":
        sidebar_html = render_to_string('data_processing_project.html',request=request)
    elif project_type == "Cash Forecasting":
        sidebar_html = render_to_string('cash_forecasting_project.html',request=request)
    else:
        sidebar_html = render_to_string('image_processing_project.html', request=request)

    return JsonResponse({'sidebar_html': sidebar_html, 'p_type': project_type})



@csrf_exempt
def right_side_bar(request):
    project_id = request.POST['project_id']
    data_base = SparkDataBase()
    project_type = data_base.retreive_project_type(project_id)
    if project_type == 'Data Processing':
        data_frame = data_base.find_data_frame(int(project_id))
        row = data_frame.shape[0]
        col = data_frame.shape[1]
        features_dtypes = data_frame.dtypes.apply(lambda feature: feature.name).to_dict()
        dtypes = data_frame.dtypes.value_counts()
        memory = data_frame.memory_usage().sum()
        memory = memory*0.001
        memory = round(memory, 2)
        saved_models_accuracy = []
        saved_models_info = data_base.load_models_accuracy(int(project_id))
        if saved_models_info == 0:
            saved_models_info_html = "<p>No models are saved for this project</p>"
        else:
            for each in saved_models_info:
                new = {}
                try:
                    new['accuracy'] = round(each['accuracy']*100,2)
                    new['algorithm'] = each['algorithm']
                    saved_models_accuracy.append(new)
                except KeyError:
                    continue
            if len(saved_models_accuracy) == 0:
                saved_models_info_html = "<p>No models are saved for this project</p>"
            else:
                context = {"savedmodels": saved_models_accuracy}
                saved_models_info_html = render_to_string("Controls/saved_models_info.html",
                                                        context=context,request=request)

        saved_cluster_models = []
        saved_cluster_models_from_db = data_base.load_models_cluster(project_id)
        if saved_cluster_models_from_db == 0:
            saved_cluster_models_info_html = "<p>No models are saved for this project</p>"
        else:
            for each in saved_cluster_models_from_db:
                new = {}
                try:
                    new['algorithm'] = each['algorithm']
                    new['k_cluster'] = each['k_cluster']
                    saved_cluster_models.append(new)
                except:
                    continue
            if len(saved_cluster_models) == 0:
                saved_cluster_models_info_html = "<p>No models are saved for this project</p>"
            else:
                context = {"saved_cluster_models": saved_cluster_models}
                saved_cluster_models_info_html = render_to_string("Controls/saved_cluster_models.html",
                                                                  context=context, request=request)

        saved_regression_models = []
        saved_regression_models_from_db = data_base.load_models_regression_views(project_id)
        if saved_regression_models_from_db == 0:
            saved_regression_models_info_html = "<p>No models are saved for this project</p>"
        else:
            for each in saved_regression_models_from_db:
                new = {}
                try:
                    new['algorithm'] = each['algorithm']
                    new['train_Rsq'] = round(each['train_Rsq']*100, 2)
                    new['train_adjRsq'] = round(each['train_adjRsq']*100, 2)
                    saved_regression_models.append(new)
                except:
                    continue
        if len(saved_regression_models) == 0:
            saved_regression_models_info_html = "<p>No models are saved for this project</p>"
        else:
            context = {"saved_regression_models": saved_regression_models}
            saved_regression_models_info_html = render_to_string("Controls/saved_regression_models.html",
                                                                 context=context, request=request)

        context = {'row': row, 'col': col, 'features_dtypes': features_dtypes, 'dtypes': dtypes, 'memory': memory}
        right_side_bar_html = render_to_string('Controls/rightsidebar.html', context=context, request=request)
        return JsonResponse({'project_type':project_type,
                             'right_side_bar_html': right_side_bar_html,
                             'saved_models_info_html': saved_models_info_html,
                             'saved_cluster_models_info_html': saved_cluster_models_info_html,
                             'saved_regression_models_info_html': saved_regression_models_info_html})

    elif project_type == 'Cash Forecasting':
        data_frame = data_base.find_data_frame(int(project_id))
        all_be_name_list = list(sorted(set(data_frame.billing_entity)))
        # dic = create_dic(data_frame)
        # be_name = list(dic.keys())
        row = data_frame.shape[0]
        col = data_frame.shape[1]
        features_dtypes = data_frame.dtypes.apply(lambda feature: feature.name).to_dict()
        dtypes = data_frame.dtypes.value_counts()
        memory = data_frame.memory_usage().sum()
        memory = memory * 0.001
        memory = round(memory, 2)
        context1 = {'row': row, 'col': col, 'features_dtypes': features_dtypes, 'dtypes': dtypes, 'memory': memory}
        context2 = {'be_name': all_be_name_list}
        right_side_bar_html = render_to_string('Controls/cash_rightsidebar.html', context=context1, request=request)
        billing_entity_html = render_to_string('Cash_Forecasting/billing_entity.html', context=context2, request=request)
        return JsonResponse({'project_type': project_type,
                             'right_side_bar_html': right_side_bar_html,
                             'billing_entity_html': billing_entity_html})

    else:
        labels = []
        acc = 0
        saved_model = 0
        category = get_categories_path(project_id)
        for each in category:
            labels.append(each.split("/")[-1])
        directory = r"mlp/SaveTrainDataModels/" + str(project_id) + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = "train_" + str(project_id) + ".pickle"
        with open(directory + filename, 'rb') as f:
            x_train, x_test, y_train, y_test = pickle.load(f)
        if os.path.exists(directory + str(project_id) + ".h5"):
            graph = Graph()
            tf.reset_default_graph()
            with graph.as_default():
                sess = Session()
                with sess.as_default():
                    model = load_model(directory + str(project_id) + ".h5")
                    accuracy = model.evaluate(x_test, y_test, verbose=1)
                    saved_model = 1
                    acc = round(accuracy[1],2)*100
        context = {'labels': labels, 'accuracy': acc}
        right_side_bar_html = render_to_string('Controls/image_right_side_bar.html', request=request, context=context)
        return JsonResponse({'project_type': project_type,
                             'right_side_bar_html': right_side_bar_html, 'saved_model': saved_model})



@csrf_exempt
def summary(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    numerical_summary_html = obj_exp_data_analysis.numerical_summary_from_data_frame()
    categorical_summary_html = obj_exp_data_analysis.categorical_summary_from_data_frame()
    summary_html = render_to_string('summary.html', request=request)
    if categorical_summary_html is None and numerical_summary_html is None:
        return JsonResponse({'categorical_summary_html': '<p>No categorical features in the data</p>',
                             'summary_html': summary_html,
                             'numerical_summary_html': '<p>No numerical features in the data</p>'})
    elif categorical_summary_html is None:
        return JsonResponse({'numerical_summary_html': numerical_summary_html, 'summary_html': summary_html,
                             'categorical_summary_html': '<p>No categorical features in the data</p>'})
    elif numerical_summary_html is None:
        return JsonResponse({'categorical_summary_html': categorical_summary_html, 'summary_html': summary_html,
                             'numerical_summary_html': '<p>No numerical features in the data</p>'})
    else:
        return JsonResponse({'numerical_summary_html': numerical_summary_html,
                             'categorical_summary_html': categorical_summary_html,
                             'summary_html': summary_html})


@csrf_exempt
def plotting(request):
    plotting_html = render_to_string('Charts/plotting.html', request=request)
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    numerical_features = obj_exp_data_analysis.numerical_features()
    context = {'features': numerical_features}
    distribution_plot_html = obj_exp_data_analysis.distribution_plot(numerical_features[0])
    box_plot_html = obj_exp_data_analysis.box_plot(numerical_features[0], numerical_features[1])
    scatter_plot_html = obj_exp_data_analysis.scatter_plot(numerical_features[0], numerical_features[1])
    correlation_plot_html = obj_exp_data_analysis.correlation_plot()
    choose_features_distribution_html = render_to_string("Charts/distributionplot.html", context=context, request=request)
    features = obj_exp_data_analysis.features_list()
    features.remove(numerical_features[1])
    features.insert(0, numerical_features[1])
    context = {'features': numerical_features, 'features1': features}
    choose_features_scatter_html = render_to_string("Charts/scatterplot.html", context=context, request=request)
    choose_features_box_html = render_to_string("Charts/boxplot.html", context=context, request=request)
    return JsonResponse({'plotting_html': plotting_html, 'choose_features_distribution_html': choose_features_distribution_html,
                         'distribution_plot_html': distribution_plot_html, 'choose_features_scatter_html': choose_features_scatter_html,
                         'scatter_plot_html': scatter_plot_html,'choose_features_box_html': choose_features_box_html,
                         'box_plot_html': box_plot_html, 'correlation_plot_html': correlation_plot_html,
                         'correlation_plot_heading': '<label>Correlation Plot</label>'
                         })


@csrf_exempt
def distribution_plot(request):
    feature = request.POST['col']
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    distribution_plot_html = obj_exp_data_analysis.distribution_plot(feature)
    return JsonResponse({'distribution_plot_html': distribution_plot_html})


@csrf_exempt
def scatter_plot(request):
    feature1 = request.POST['col1']
    feature2 = request.POST['col2']
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    scatter_plot_html = obj_exp_data_analysis.scatter_plot(feature1, feature2)
    return JsonResponse({'scatter_plot_html': scatter_plot_html})


@csrf_exempt
def box_plot(request):
    feature1 = request.POST['col1']
    feature2 = request.POST['col2']
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    box_plot_html = obj_exp_data_analysis.box_plot(feature1, feature2)
    return JsonResponse({'box_plot_html': box_plot_html})


@csrf_exempt
def missing_values(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    miss_data = data_frame.isnull().sum().to_dict()
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    numerical_features = obj_exp_data_analysis.numerical_features()
    for key, value in list(miss_data.items()):
        if value == 0:
            del miss_data[key]
    if not bool(miss_data):
        return JsonResponse(
            {'missing_value_html': "<div class='row'><div class='col-lg-8'><div class='card'>"
                                   "<div class='card-header'><label>Missing Values Imputation</label>"
                                   "</div><div class='card-body'>No features have missing values</div>"
                                   "</div></div></div>"})
    else:
        table_data = []
        for key, value in list(miss_data.items()):
            d = {}
            d['col'] = key
            if key in numerical_features:
                d['type'] = 'Numerical'
                d['replace'] = ['mean', 'median', 'mode', 'value', 'delete']
            else:
                d['type'] = 'Categorical'
                d['replace'] = ['mode', 'delete']
            d['count'] = value
            table_data.append(d)
        context = {'tabledata': table_data}
        missing_value_html = render_to_string('DataPreProcessing/missingvalues.html', context=context, request=request)
        return JsonResponse({'missing_value_html': missing_value_html})

@csrf_exempt
def missing_value_impuation(request):
    table_data_as_string = request.POST['tabledata']
    table_data_as_list = json.loads(table_data_as_string)
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    missing_value_operations = MissingValues(data_frame)
    updated_data_frame = missing_value_operations.missing_value_methods(table_data_as_list)
    if updated_data_frame is not None:
        data_base = SparkDataBase()
        data_base.update_document(updated_data_frame,project_id)
    missing_data_frame = data_frame.isnull().sum()
    miss_data = missing_data_frame.to_dict()
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    numerical_features = obj_exp_data_analysis.numerical_features()
    for key, value in list(miss_data.items()):
        if value == 0:
            del miss_data[key]
    if not bool(miss_data):
        return JsonResponse(
            {'missing_value_html': "<div class='row'><div class='col-lg-8'><div class='card'>"
                                   "<div class='card-header'><label>Missing Values Imputation</label></div>"
                                   "<div class='card-body'>No features have missing values</div></div></div></div>"})
    else:
        table_data = []
        for key, value in list(miss_data.items()):
            d = {}
            d['col'] = key
            if key in numerical_features:
                d['type'] = 'Numerical'
                d['replace'] = ['mean', 'median', 'mode', 'value', 'delete']
            else:
                d['type'] = 'Categorical'
                d['replace'] = ['mode', 'delete']
            d['count'] = value
            table_data.append(d)
        context = {'tabledata': table_data}
        missing_value_html = render_to_string('DataPreProcessing/missingvalues.html', context=context, request=request)
        return JsonResponse({'missing_value_html': missing_value_html})


@csrf_exempt
def normalization(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    features = obj_exp_data_analysis.numerical_features()
    methods = ['Normalization in range', "standard Normalization"]
    table_data = []
    for each in features:
        d = {}
        d['feature'] = each
        d['min'] = data_frame[each].min()
        d['max'] = data_frame[each].max()
        d['action'] = methods
        table_data.append(d)
    context = {'table_data': table_data}
    normalization_html = render_to_string("DataPreProcessing/normalization.html", context=context, request=request)
    distribution_plot_html = obj_exp_data_analysis.distribution_plot(features[0])
    context = {'features': features, 'id': "distribution_plot_normalization",
               "function": "distribution_plot_normalization()"}
    distribution_plot_choose_features_html = render_to_string('Charts/user_visualization.html', context=context,
                                                              request=request)
    return JsonResponse({'normalization_html': normalization_html, 'distribution_plot_html': distribution_plot_html,
                         'distribution_plot_choose_features_html': distribution_plot_choose_features_html})

@csrf_exempt
def normalization_methods(request):
    table_data_as_string = request.POST['tabledata']
    table_data_as_list = json.loads(table_data_as_string)
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    normalization_of_data = Normalization(data_frame)
    updated_data_frame = normalization_of_data.normalization_methods(table_data_as_list)
    if updated_data_frame is not None:
        data_base = SparkDataBase()
        data_base.update_document(updated_data_frame, project_id)
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    numerical_features = obj_exp_data_analysis.numerical_features()
    methods = ['Normalization in range', "standard Normalization"]
    table_data = []
    for each in numerical_features:
        d = {}
        d['feature'] = each
        d['min'] = data_frame[each].min()
        d['max'] = data_frame[each].max()
        d['action'] = methods
        table_data.append(d)
    context = {'table_data': table_data}
    normalization_html = render_to_string("DataPreProcessing/normalization.html", context=context, request=request)
    context = {'features': numerical_features, 'id': "distribution_plot_normalization",
               "function": "distribution_plot_normalization()"}
    distribution_plot_html = obj_exp_data_analysis.distribution_plot(numerical_features[0])
    distribution_plot_choose_features_html = render_to_string('Charts/user_visualization.html', context=context,
                                                              request=request)
    return JsonResponse({'normalization_html': normalization_html, 'distribution_plot_html': distribution_plot_html,
                         'distribution_plot_choose_features_html': distribution_plot_choose_features_html})


@csrf_exempt
def distribution_plot_normalization(request):
    project_id = int(request.POST['project_id'])
    feature = request.POST['feature']
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    distribution_plot_html = obj_exp_data_analysis.distribution_plot(feature)
    return JsonResponse({'distribution_plot_html': distribution_plot_html})


@csrf_exempt
def deletefeature(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    features = obj_exp_data_analysis.features_list()
    context = {'features': features}
    delete_feature_html = render_to_string('DataPreProcessing/Deletefeatures.html', context=context, request=request)
    return JsonResponse({'delete_feature_html': delete_feature_html})


@csrf_exempt
def delete_features_handling(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    selected_features_as_string = request.POST['tabledata']
    selected_features_as_list = json.loads(selected_features_as_string)
    for feature in selected_features_as_list:
        data_frame.drop(feature, axis=1, inplace=True)
    data_base = SparkDataBase()
    data_base.update_document(data_frame, project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    features = obj_exp_data_analysis.features_list()
    context = {'features': features}
    delete_feature_html = render_to_string('DataPreProcessing/Deletefeatures.html', context=context, request=request)
    return JsonResponse({'delete_feature_html': delete_feature_html})


@csrf_exempt
def outliers(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    numerical_data_frame = data_frame.select_dtypes(exclude=['object'])
    Q1 = numerical_data_frame.quantile(0.25)
    Q3 = numerical_data_frame.quantile(0.75)
    IQR = Q3 - Q1
    outliers_summary = ((numerical_data_frame < (Q1 - 1.5 * IQR)) | (numerical_data_frame > (Q3 + 1.5 * IQR))).sum()
    outliers_summary = outliers_summary.to_dict()
    context = {'table_data': outliers_summary}
    outliers_html = render_to_string('DataPreProcessing/outliers.html', context=context, request=request)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    features = obj_exp_data_analysis.numerical_features()
    box_plot_html = obj_exp_data_analysis.box_plot_outliers(feature=features[0])
    context = {'features': features, 'id': 'boxplotoutlierfeature', 'function': 'box_plot_outliers()'}
    box_plot_choose_features_html = render_to_string('Charts/user_visualization.html', context=context, request=request)
    return JsonResponse({'outliers_html': outliers_html, 'box_plot_html': box_plot_html,
                         'box_plot_choose_features_html': box_plot_choose_features_html})


@csrf_exempt
def outliers_handling(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    numerical_data_frame = data_frame.select_dtypes(exclude=['object'])
    selected_features_as_string = request.POST['tabledata']
    selected_features_as_list = json.loads(selected_features_as_string)
    Q1 = numerical_data_frame.quantile(0.25)
    Q3 = numerical_data_frame.quantile(0.75)
    IQR = Q3 - Q1
    for feature in selected_features_as_list:
        data_frame[feature] = numerical_data_frame[
            ~((numerical_data_frame[[feature]] < (Q1 - 1.5 * IQR)) | (numerical_data_frame[[feature]] > (Q3 + 1.5 * IQR))).any(axis=1)]
    data_base = SparkDataBase()
    data_base.update_document(data_frame, project_id)
    numerical_data_frame = data_frame.select_dtypes(exclude=['object'])
    Q1 = numerical_data_frame.quantile(0.25)
    Q3 = numerical_data_frame.quantile(0.75)
    IQR = Q3 - Q1
    outliers_summary = ((numerical_data_frame < (Q1 - 1.5 * IQR)) | (numerical_data_frame > (Q3 + 1.5 * IQR))).sum()
    outliers_summary = outliers_summary.to_dict()
    context = {'table_data': outliers_summary}
    outliers_html = render_to_string('DataPreProcessing/outliers.html', context=context, request=request)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    features = obj_exp_data_analysis.numerical_features()
    box_plot_html = obj_exp_data_analysis.box_plot_outliers(feature=features[0])
    context = {'features': features, 'id': 'boxplotoutlierfeature', 'function': 'box_plot_outliers()'}
    box_plot_choose_features_html = render_to_string('Charts/user_visualization.html', context=context, request=request)
    return JsonResponse({'outliers_html': outliers_html, 'box_plot_html': box_plot_html,
                         'box_plot_choose_features_html': box_plot_choose_features_html})


@csrf_exempt
def box_plot_outliers(request):
    project_id = int(request.POST['project_id'])
    feature = request.POST['feature']
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    box_plot_html = obj_exp_data_analysis.box_plot_outliers(feature)
    return JsonResponse({'box_plot_html': box_plot_html})


@csrf_exempt
def encoding(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    features = obj_exp_data_analysis.categorical_features()
    methods = ['Label Encoding', 'One Hot Encoding']
    if len(features) != 0:
        context = {'features': features, 'methods': methods}
        encoder_html = render_to_string('DataPreProcessing/encoding.html', context=context, request=request)
        return JsonResponse({'encoder_html': encoder_html})
    else:
        return JsonResponse({'encoder_html':"<div class='row'><div class='col-lg-8'><div class='card'>"
                            "<div class='card-header'><label>Encoding of Features</label></div>"
                             "<div class='card-body'>No categorical features present in the data </div></div></div></div>"})


@csrf_exempt
def encode_method(request):
    project_id = int(request.POST['project_id'])
    method = request.POST['method']
    selected_features = json.loads(request.POST['selected_features'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_encoding = Encoder(data_frame)
    if method == "Label Encoding":
        data_frame = obj_encoding.label_encoding(selected_features)
    else:
        data_frame = obj_encoding.one_hot_encoding(selected_features)
    data_base = SparkDataBase()
    data_base.update_document(data_frame, project_id)
    return JsonResponse({'encoded_html': '<h4>Features are encoded</h4>'})


@csrf_exempt
def sampling(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    methods = ['Under Sampling', 'Over Sampling', 'SMOTE', 'Tomek']
    features = obj_exp_data_analysis.features_list()
    context = {'features': features, 'methods': methods, 'id': 'sampling_feature', 'method_id': 'sampling_method'}
    sampling_html = render_to_string('DataPreProcessing/sampling.html', context=context, request=request)
    return JsonResponse({'sampling_html': sampling_html})


@csrf_exempt
def sampling_info(request):
    project_id = int(request.POST['project_id'])
    feature = request.POST['feature']
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_sampling = Sampling()
    table_data, plot_html = obj_sampling.target_info(data_frame, feature)
    context = {'table_data': table_data, 'feature': feature}
    sample_info_html = render_to_string("DataPreProcessing/sample_table_info.html", context=context, request=request)
    return JsonResponse({'sample_info_html': sample_info_html, 'sample_bar_plot': plot_html})


# Return Codes :
# 0 - Response if dataset is perfect
# 1 - Response if Catagorical data is found on the dataset for SMOTE
# 2 - Response if values in a Class are not enough for SMOTE
# 3 - Response if Tomek cant be performed

@csrf_exempt
def sampling_methods(request):
    project_id = int(request.POST['project_id'])
    feature = request.POST['feature']
    method = request.POST['method']
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_sampling = Sampling()
    if method == "Under Sampling":
        data_frame = obj_sampling.under_sampling(data_frame, feature)
    elif method == "Over Sampling":
        data_frame = obj_sampling.over_sampling(data_frame, feature)
    elif method == "SMOTE":
        flag = obj_sampling.smote_validate(data_frame, feature)
        if flag == 1 or flag == 2:
            table_data, plot_html = obj_sampling.target_info(data_frame, feature)
            context = {'table_data': table_data, 'feature': feature}
            sample_info_html = render_to_string("DataPreProcessing/sample_table_info.html", context=context,request=request)
            return JsonResponse({'sample_info_html': sample_info_html, 'sample_bar_plot': plot_html, 'error_info': flag })
        elif flag == 0:
            flag = obj_sampling.smote(data_frame, feature)
            #flag = data_frame
            if type(flag) == int:
                table_data, plot_html = obj_sampling.target_info(data_frame, feature)
                context = {'table_data': table_data, 'feature': feature}
                sample_info_html = render_to_string("DataPreProcessing/sample_table_info.html", context=context,
                                                    request=request)
                return JsonResponse(
                    {'sample_info_html': sample_info_html, 'sample_bar_plot': plot_html, 'error_info': flag})
            else:
                data_frame = flag

    elif method == "Tomek":
        flag = obj_sampling.tomek(data_frame, feature)     # Returns 3 if Tomek is not performed. Returns dataframe if Tomek is performed
        if type(flag) == int:  # return type check if Tomek is performed or no
            table_data, plot_html = obj_sampling.target_info(data_frame, feature)
            context = {'table_data': table_data, 'feature': feature}
            sample_info_html = render_to_string("DataPreProcessing/sample_table_info.html", context=context, request=request)
            return JsonResponse({'sample_info_html': sample_info_html, 'sample_bar_plot': plot_html, 'error_info': flag})
        else:
            data_frame = flag  # Returned Dataframe is copied from flag to data_frame

    data_base.update_document(data_frame, project_id)
    table_data, plot_html = obj_sampling.target_info(data_frame, feature)
    context = {'table_data': table_data, 'feature': feature}
    sample_info_html = render_to_string("DataPreProcessing/sample_table_info.html", context=context, request=request)
    return JsonResponse({'sample_info_html': sample_info_html, 'sample_bar_plot': plot_html})


@csrf_exempt
def modelling(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    features = obj_exp_data_analysis.features_list()
    context = {'features': features}
    modelling_html = render_to_string('Development/modelling.html', context=context, request=request)
    database = SparkDataBase()
    target_feature = None
    saved_models_from_database = database.load_models_accuracy(project_id)
    if saved_models_from_database == 0:
        saved_models_from_database = 0
    else:
        for each in saved_models_from_database:
            unique_model_id = ""
            for item in each['algorithm'].split(" "):
                unique_model_id += item[0]
            unique_model_id += str(project_id)
            api_url = database.retrive_models_api_url(unique_model_id)
            each['api_url'] = api_url
        target_feature = database.target_feature(project_id)
    return JsonResponse({'modelling_html': modelling_html,'saved_models':saved_models_from_database,'target_feature': target_feature})


@csrf_exempt
def type_of_algorithms(request):
    project_id=int(request.POST["project_id"])
    algorithms = ['K Nearest Neighbour', 'Decision Tree', 'Random Forest', 'Support Vector Machine', 'XgBoost']
    database=SparkDataBase()
    saved_models_from_db = database.load_models_accuracy(project_id)
    if saved_models_from_db != 0:
        saved_models = []
        for each in saved_models_from_db:
            saved_models.append(each['algorithm'])
        for each in saved_models:
            algorithms.remove(each)
    context = {'algorithms': algorithms}
    models_list_html = render_to_string('Development/modelslist.html', context=context, request=request)
    return JsonResponse({'models_list_html': models_list_html})


@csrf_exempt
def selected_algorithm_input_parameters(request):
    selected_algorithm = request.POST["selected_algorithm"]
    data_base = SparkDataBase()
    input_parameters = data_base.default_input_parameters(selected_algorithm)
    return JsonResponse({'input_parameters': input_parameters})

@csrf_exempt
def edit_algorithm_input_parameters(request):
    selected_algorithm = request.POST["selected_algorithm"]
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    algorithm_input_parameters = data_base.default_input_parameters(selected_algorithm)
    previous_input_parameters = data_base.load_models_input_parameters(project_id, selected_algorithm)
    i = 0
    for each in algorithm_input_parameters:
        if each[1] == "input":
            each[3] = previous_input_parameters[i]
        else:
            each[3].remove(previous_input_parameters[i])
            each[3].insert(0, previous_input_parameters[i])
        i += 1
    input_parameters = algorithm_input_parameters
    return JsonResponse({'input_parameters': input_parameters})


@csrf_exempt
def algorithms(request):
    project_id = int(request.POST['project_id'])
    data_options_string = request.POST['data_options']
    data_options = json.loads(data_options_string)
    algorithm_info_string = request.POST['algorithm_info']
    algorithm_info = json.loads(algorithm_info_string)
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    execute_algorithms = Algorithms(data_frame, project_id)
    data_result = execute_algorithms.data_processing(data_options, algorithm_info)
    return JsonResponse({'data_result': data_result})


@csrf_exempt
def current_data_frame_one_step_back(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_base.update_current_data_frame_with_one_step_back(project_id)
    return JsonResponse({'info': "success"})

@csrf_exempt
def save_input_parameters(request):
    project_id = int(request.POST['project_id'])
    algorithm=request.POST["algorithm"]
    input_parameters_string = request.POST['input_parameters']
    input_parameters = json.loads(input_parameters_string)
    database=SparkDataBase()
    database.save_input_parameters(project_id, algorithm, input_parameters)
    return JsonResponse({'success': "success"})

@csrf_exempt
def validation(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    algorithms = data_base.saved_model_for_project(project_id)
    context = {'algorithms': algorithms}
    validation_html = render_to_string('Development/validations.html', context=context, request=request)
    return JsonResponse({'validation_html': validation_html})


@csrf_exempt
def validation_data(request):
    project_id = int(request.POST['project_id'])
    algorithm = request.POST["algorithm"]
    data_base = SparkDataBase()
    target_feature = data_base.target_feature(project_id)
    data_frame = data_base.find_data_frame(project_id)
    input_features = list(data_frame)
    input_features.remove(target_feature)
    context = {'features': input_features, 'output_feature': target_feature}
    validation_data_html = render_to_string("Development/validation_data.html", context=context, request=request)
    return JsonResponse({'validation_data_html': validation_data_html})


@csrf_exempt
def predict_output(request):
    project_id = int(request.POST['project_id'])
    algorithm = request.POST["algorithm"]
    input_data = json.loads(request.POST['input_data'])
    predict_data = {}
    for feature in input_data:
        try:
            predict_data[feature[0]] = int(feature[1])
        except:
            predict_data[feature[0]] = float(feature[1])

    predict_data = pd.DataFrame(predict_data, index=[0])
    data_base = SparkDataBase()
    model = data_base.load_model(project_id, algorithm)
    predicted = execute_selected_model(model, predict_data)
    predicted = predicted.tolist()
    return JsonResponse({'predicted_result': predicted})

@csrf_exempt
def save_model_for_api(request):
    project_id = int(request.POST["project_id"])
    algorithm = request.POST["algorithm"]
    data_base = SparkDataBase()
    unique_model_id = ""
    for each in algorithm.split(" "):
        unique_model_id += each[0]
    unique_model_id += str(project_id)
    api_url = 'http://10.182.235.159:8080/api/'+unique_model_id+'/'
    data_base.save_model_for_api(project_id,unique_model_id,api_url)
    return JsonResponse({'api_url': api_url})

@csrf_exempt
#Loads Binning Html page
def binning(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    features = obj_exp_data_analysis.numerical_features() #returns features which are of numeric type
    action = ['Binning by Width', 'Binning by Frequency']
    option = ['Auto Labeling', 'Manual Labeling']
    table_data = []
    for each in features:
        d = {}
        d['feature'] = each
        d['action'] = action
        d['option'] = option
        table_data.append(d)
    context = {'table_data': table_data} #passes table_data as context for the HTML page to render
    binning_html = render_to_string("DataPreProcessing/binning.html", context=context, request=request)
    return JsonResponse({'binning_html': binning_html})

@csrf_exempt
#Handles Binning Process and reloads the binning page once the binning process is complete
def binning_handler(request):
    table_data_as_string = request.POST['tabledata']    #'tabledata' is recieved from binningsubmit method from project.js
    table_data_as_list = json.loads(table_data_as_string)
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    binning_of_data = Binning(data_frame)
    updated_data_frame = binning_of_data.binning_methods(table_data_as_list)
    if updated_data_frame is not None: #loads updated dataframe
        data_base.update_document(updated_data_frame, project_id)

    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    features = obj_exp_data_analysis.numerical_features()   #returns features which are of numeric type
    action = ['Binning by Width', 'Binning by Frequency']
    option = ['Auto Labeling', 'Manual Labeling']
    table_data = []
    for each in features:
        d = {}
        d['feature'] = each
        d['action'] = action
        d['option'] = option
        table_data.append(d)
    context = {'table_data': table_data}    #passes table_data as context for the HTML page to render
    binning_html = render_to_string("DataPreProcessing/binning.html", context=context, request=request)
    return JsonResponse({'binning_html': binning_html})

@csrf_exempt
#Displays all the data on View data page
def view_All_Data(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    dataframe_html = data_frame.to_html(classes=["table", "table-responsive", "table-bordered",  "table-hover"])
    return JsonResponse({'dataframe_html':dataframe_html})

@csrf_exempt
def clustering(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    algorithms = ['KMeans']

    saved_cluster_models_from_db = data_base.load_models_cluster(project_id)
    if saved_cluster_models_from_db != 0:
        saved_models = []
        for each in saved_cluster_models_from_db:
            saved_models.append(each['algorithm'])
            unique_model_id = ""
            for item in each['algorithm'].split(" "):
                unique_model_id += item
            unique_model_id += str(project_id)
            api_url = data_base.cluster_retrive_models_api_url(unique_model_id)
            each['api_url'] = api_url

        for each in saved_models:
            algorithms.remove(each)

    context = {'algorithms': algorithms}
    clustering_html = render_to_string('Development/clustering.html', request=request)
    models_list_html = render_to_string('Development/cluster_model_list.html', context=context, request=request)
    return JsonResponse({'clustering_html': clustering_html, 'models_list_html': models_list_html, 'saved_cluster_models':saved_cluster_models_from_db})


@csrf_exempt
def selected_cluster_algorithm_input_parameters(request):
    project_id = int(request.POST['project_id'])
    selected_algorithm = request.POST["selected_algorithm"]
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    features = obj_exp_data_analysis.categorical_features()
    cluster_index = 'Cluster_Labels' in features

    if cluster_index:
        data_frame = data_frame.drop('Cluster_Labels', axis=1)
        features.remove('Cluster_Labels')

    if len(features) > 0:
        data_result = 0
        return JsonResponse({'data_result': data_result})
    if selected_algorithm == 'KMeans':
        execute_algorithms = Clustering(data_frame, project_id)
        elbow_html = execute_algorithms.elbow_method()
    else:
        elbow_html = 0

    input_parameters = data_base.cluster_input_parameters(selected_algorithm)
    return JsonResponse({'input_parameters': input_parameters, 'elbow_plot_points': elbow_html})

@csrf_exempt
def save_cluster_input_parameters(request):
    project_id = int(request.POST['project_id'])
    algorithm=request.POST["algorithm"]
    input_parameters_string = request.POST['input_parameters']
    input_parameters = json.loads(input_parameters_string)
    database = SparkDataBase()
    database.save_cluster_input_parameters(project_id, algorithm, input_parameters)
    return JsonResponse({'success': "success"})

@csrf_exempt
def edit_cluster_algorithm_input_parameters(request):
    selected_algorithm = request.POST["selected_algorithm"]
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    algorithm_input_parameters = data_base.cluster_input_parameters(selected_algorithm)
    previous_input_parameters = data_base.load_cluster_models_input_parameters(project_id, selected_algorithm)
    i = 0
    for each in algorithm_input_parameters:
        if each[1] == "input":
            each[3] = previous_input_parameters[i]
        else:
            each[3].remove(previous_input_parameters[i])
            each[3].insert(0, previous_input_parameters[i])
        i += 1
    input_parameters = algorithm_input_parameters

    if selected_algorithm == 'KMeans':
        execute_algorithms = Clustering(data_frame, project_id)
        elbow_html = execute_algorithms.elbow_method()
    else:
        elbow_html = 0


    return JsonResponse({'input_parameters': input_parameters, 'elbow_plot_points': elbow_html})


@csrf_exempt
def cluster_algorithms(request):
    project_id = int(request.POST['project_id'])
    algorithm_info_string = request.POST['algorithm_info']
    algorithm_info = json.loads(algorithm_info_string)
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    features = obj_exp_data_analysis.categorical_features()
    cluster_index = 'Cluster_Labels' in features

    if cluster_index:
        data_frame = data_frame.drop('Cluster_Labels', axis=1)
        features.remove('Cluster_Labels')

    execute_algorithms = Clustering(data_frame, project_id)
    data_result, plot, updated_data_frame = execute_algorithms.data_cluster_processing(algorithm_info)
    data_base.update_document(updated_data_frame, project_id)

    return JsonResponse({'data_result': data_result, 'plot_points':plot})

@csrf_exempt
def cluster_validation(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    algorithms = data_base.saved_cluster_model_for_project(project_id)
    context = {'algorithms': algorithms}
    validation_html = render_to_string('Development/cluster_validation.html', context=context, request=request)
    return JsonResponse({'validation_html': validation_html})

@csrf_exempt
def cluster_validation_data(request):
    project_id = int(request.POST['project_id'])
    algorithm = request.POST["algorithm"]
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    input_features = list(data_frame)

    cluster_index = 'Cluster_Labels' in input_features
    if cluster_index:
        input_features.remove('Cluster_Labels')

    target_feature = 'Cluster_Labels'
    context = {'features': input_features, 'output_feature': target_feature}
    validation_data_html = render_to_string("Development/cluster_validation_data.html", context=context, request=request)
    return JsonResponse({'validation_data_html': validation_data_html})

@csrf_exempt
def cluster_predict_output(request):
    project_id = int(request.POST['project_id'])
    algorithm = request.POST["algorithm"]
    input_data = json.loads(request.POST['input_data'])
    predict_data = {}
    for feature in input_data:
        try:
            predict_data[feature[0]] = int(feature[1])
        except:
            predict_data[feature[0]] = float(feature[1])

    predict_data = pd.DataFrame(predict_data, index=[0])
    data_base = SparkDataBase()
    model = data_base.load_cluster_model(project_id, algorithm)
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    execute_algorithms = Clustering(data_frame, project_id)
    predicted = execute_algorithms.cluster_execute_selected_model(model, predict_data)
    return JsonResponse({'predicted_result': predicted})

@csrf_exempt
def save_cluster_model_for_api(request):
    project_id = int(request.POST["project_id"])
    algorithm = request.POST["algorithm"]
    data_base = SparkDataBase()
    unique_model_id = ""
    for each in algorithm.split(" "):
        unique_model_id += each
    unique_model_id += str(project_id)
    api_url = 'http://10.182.235.159:8080/api/clustering/'+unique_model_id+'/'
    data_base.cluster_save_model_for_api(project_id, unique_model_id, api_url)
    return JsonResponse({'api_url': api_url})

@csrf_exempt
def plot_normal_distribution(request):
    plotting_html = render_to_string('DataPreProcessing/normalDistribution.html', request=request)
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    numerical_features = obj_exp_data_analysis.numerical_features()
    context = {'features': numerical_features}
    normal_distribution_plot_html = obj_exp_data_analysis.normal_distribution_plot(numerical_features[0])
    choose_features_normal_distribution_html = render_to_string("DataPreProcessing/normalPlot.html", context=context,
                                                                request=request)
    return JsonResponse({'normal_distribution_plot_html': normal_distribution_plot_html,
                         'choose_features_normal_distribution_html': choose_features_normal_distribution_html,
                         'plotting_html': plotting_html})

@csrf_exempt
def normal_distribution(request):
    feature = request.POST['col']
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    normal_distribution_plot_html = obj_exp_data_analysis.normal_distribution_plot(feature)
    return JsonResponse({'normal_distribution_plot_html': normal_distribution_plot_html})

@csrf_exempt
def distribute_normal(request):
    feature = request.POST['col']
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    data_frame, val = obj_exp_data_analysis.normal_distribution(feature)
    data_base.update_document(data_frame, project_id)
    plotting_html = render_to_string('DataPreProcessing/normalDistribution.html', request=request)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    numerical_features = obj_exp_data_analysis.numerical_features()
    context = {'features': numerical_features}
    normal_distribution_plot_html = obj_exp_data_analysis.normal_distribution_plot(feature)
    choose_features_normal_distribution_html = render_to_string("DataPreProcessing/normalPlot.html", context=context,
                                                                request=request)
    return JsonResponse({'normal_distribution_plot_html': normal_distribution_plot_html,
                         'choose_features_normal_distribution_html': choose_features_normal_distribution_html,
                         'plotting_html': plotting_html})

@csrf_exempt
def Regression_load(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    features = obj_exp_data_analysis.features_list()
    algorithm = ['Linear_Regression']


    saved_regression_models_from_db = data_base.load_models_regression_views(project_id)
    if saved_regression_models_from_db != 0:
        saved_models = []
        for each in saved_regression_models_from_db:
            saved_models.append(each['algorithm'])
            unique_model_id = ""
            for item in each['algorithm'].split(" "):
                unique_model_id += item
            unique_model_id += str(project_id)
            api_url = data_base.regression_retrive_models_api_url(unique_model_id)
            each['api_url'] = api_url

        for each in saved_models:
            algorithm.remove(each)

        target_feature = data_base.regression_target_feature(project_id)
    else:
        saved_regression_models_from_db = 0
        target_feature = 0
    context = {'features': features}
    regression_html = render_to_string('Development/regression.html', context=context, request=request)

    algorithms = {'algorithms': algorithm}
    models_list_html = render_to_string('Development/regression_model_list.html', context=algorithms, request=request)

    return JsonResponse({'regression_html': regression_html, 'models_list_html': models_list_html,
                         'saved_regression_models' :saved_regression_models_from_db, 'target_feature':target_feature})

@csrf_exempt
def Regression_algorithms(request):
    project_id = int(request.POST['project_id'])
    data_options_string = request.POST['data_options']
    data_options = json.loads(data_options_string)
    algorithm_info_string = request.POST['algorithm_info']
    algorithm_info = json.loads(algorithm_info_string)
    selected_features_string = request.POST['features']
    selected_features = json.loads(selected_features_string)

    if selected_features == None:
        selected_features = None
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    obj_exp_data_analysis = ExpDataAnalysis(data_frame)
    features = obj_exp_data_analysis.categorical_summary_from_data_frame()
    if features != None:
       data_result = "Error"
       return JsonResponse({'data_result': data_result})
    else:
        regression = Regression(data_frame, project_id)
        data_result, summary, ols_html, unchecked_features = regression.data_regression_processing(data_options, algorithm_info, selected_features)
    return JsonResponse({'data_result': data_result, 'summary': summary, 'ols_html': ols_html, 'unchecked_features': unchecked_features})


@csrf_exempt
def Regression_model_save_api(request):
    project_id = int(request.POST["project_id"])
    algorithm = request.POST["algorithm"]
    dropfeature = request.POST["dropFeatures"]
    dropfeatures = dropfeature.split(",")
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    if dropfeatures[0] != '':
        regression = Regression(data_frame, project_id)
        updated_data_frame = regression.feature_drop(dropfeatures)

        if updated_data_frame is not None: #loads updated dataframe
            data_base.update_document(updated_data_frame, project_id)

    unique_model_id = ""
    for each in algorithm.split(" "):
        unique_model_id += each
    unique_model_id += str(project_id)
    api_url = 'http://10.182.235.159:8080/api/regression/' + unique_model_id + '/'
    data_base.regression_save_model_for_api(project_id, unique_model_id, api_url)
    return JsonResponse({'api_url': api_url})

@csrf_exempt
def regression_validation(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    algorithms = data_base.retrieve_saved_regression_model_for_project(project_id)
    context = {'algorithms': algorithms}
    validation_html = render_to_string('Development/regression_validation.html', context=context, request=request)
    return JsonResponse({'validation_html': validation_html})

@csrf_exempt
def regression_validation_data(request):
    project_id = int(request.POST['project_id'])
    algorithm = request.POST["algorithm"]
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    input_features = list(data_frame)
    target_feature = data_base.regression_target_feature(project_id)
    input_features.remove(target_feature)
    context = {'features': input_features, 'output_feature': target_feature}
    validation_data_html = render_to_string("Development/regression_validation_data.html", context=context, request=request)
    return JsonResponse({'validation_data_html': validation_data_html})

@csrf_exempt
def regression_predict_output(request):
    project_id = int(request.POST['project_id'])
    algorithm = request.POST["algorithm"]
    input_data = json.loads(request.POST['input_data'])
    predict_data = {}
    for feature in input_data:
        try:
            predict_data[feature[0]] = int(feature[1])
        except:
            predict_data[feature[0]] = float(feature[1])

    predict_data = pd.DataFrame(predict_data, index=[0])
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(project_id)
    regression = Regression(data_frame, project_id)
    model = data_base.load_regression_model_api(project_id, algorithm)
    predicted = regression.regression_execute_selected_model(model, predict_data)
    return JsonResponse({'predicted_result': predicted})

@csrf_exempt
def Delete_proj(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    Delete_message = data_base.Delete_project(project_id)
    return JsonResponse({'Delete_message': Delete_message})


@csrf_exempt
def visual_handling(request):
    i_path = []
    categories = []
    tabledata = []
    l1 = ["category", "path"]
    project_id = int(request.POST['project_id'])
    img_path = visualizing(project_id)
    i_path = img_path
    for each in i_path:
        categories.append(each.split("/")[-2])
    for val in range(len(img_path)):
        l2 = []
        l2.append(categories[val])
        l2.append(img_path[val])
        d = dict(zip(l1, l2))
        tabledata.append(d)
    context = {'tabledata': tabledata}
    html = render_to_string('ImageProcessing/CnnHomePage.html', context=context, request=request)
    return JsonResponse({'I1': html})


@csrf_exempt
def model_handling(request):
    project_id = int(request.POST['project_id'])
    html = render_to_string('ImageProcessing/Model.html', request=request)
    return JsonResponse({'I2': html})


@csrf_exempt
def convolution_show_list(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    layer_name = "conv2d"
    dataframe = data_base.show_list(project_id, layer_name)
    dataframe.show()
    dataframe.drop('project_id')
    parameters = []
    conv_list = []
    conv_dict = {}
    index_no = []
    for each in dataframe.select('index_no').collect():
        index_no.append(each.index_no)
    for each in dataframe.select('parameters').collect():
        parameters.append(each.parameters)
    for i in range(len(parameters)):
        para_split = parameters[i].split(',')
        conv_dict = {'index_no': index_no[i], 'filter': para_split[0], 'kernel_size': para_split[1], 'strides': para_split[2], 'padding': para_split[3], 'activation': para_split[4]}
        conv_list.append(conv_dict)
    return JsonResponse({"params": conv_list})



@csrf_exempt
def conv2d_handling(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    filters = int(request.POST['filter'])
    kernel_size = int(request.POST['kernel'])
    stride = int(request.POST['stride'])
    padding = str(request.POST['padding']).strip()
    activation = str(request.POST['activation']).strip()
    index_no = data_base.create_index_model(project_id)
    parameters = str(filters) + "," + str(kernel_size) + "," + str(stride) + "," + padding + "," + activation
    l_name = "conv2d"
    data_base.image_process_model_func(project_id, str(index_no), l_name, parameters)
    res_df = spark.sql("SELECT * FROM image_process_model_table where project_id={0} order by index_no+0 ASC".format(project_id))
    res_df.show()
    res_df = res_df.drop('project_id')
    index_no = []
    layer_name = []
    parameters = []
    for each in res_df.select('index_no').collect():
        index_no.append(each.index_no)
    for each in res_df.select('layer_name').collect():
        layer_name.append(each.layer_name)
    for each in res_df.select('parameters').collect():
        parameters.append(each.parameters)
    img_path = convolution_function(project_id, index_no, layer_name, parameters)
    img_path = str(project_id) + "/" + img_path
    dataframe = data_base.show_list(project_id, l_name)
    dataframe.show()
    dataframe.drop('project_id')
    parameters = []
    conv_list = []
    conv_dict = {}
    layer_no = []
    for each in dataframe.select('parameters').collect():
        parameters.append(each.parameters)
    for each in dataframe.select('index_no').collect():
        layer_no.append(each.index_no)
    for i in range(len(parameters)):
        para_split = parameters[i].split(',')
        conv_dict = {'index_no': layer_no[i], 'filter': para_split[0], 'kernel_size': para_split[1], 'strides': para_split[2],
                     'padding': para_split[3], 'activation': para_split[4]}
        conv_list.append(conv_dict)
    context = {'img_name': img_path}
    html = render_to_string('ImageProcessing/filters-display.html', request=request, context=context)
    return JsonResponse({'I3': html, "params": conv_list})


@csrf_exempt
def conv2d_delete(request):
    project_id = int(request.POST['project_id'])
    layer_no = int(request.POST['layer_no'])
    a = 0
    if layer_no == 1:
        a = 1
        return JsonResponse({'success': "success", "a": a})
    else:
        spark.sql("INSERT OVERWRITE TABLE image_process_model_table SELECT * FROM image_process_model_table "
                  "WHERE project_id!={0} or index_no!='{1}'".format(project_id, layer_no))
        return JsonResponse({'success': "success", "a": a})


@csrf_exempt
def maxpool_show_list(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    layer_name = "maxpool"
    dataframe = data_base.show_list(project_id, layer_name)
    dataframe.show()
    dataframe.drop('project_id')
    parameters = []
    maxpool_list = []
    maxpool_dict = {}
    index_no = []
    for each in dataframe.select('index_no').collect():
        index_no.append(each.index_no)
    for each in dataframe.select('parameters').collect():
        parameters.append(each.parameters)
    for i in range(len(parameters)):
        para_split = parameters[i].split(',')
        maxpool_dict = {'index_no': index_no[i], 'pool_size': para_split[0], 'strides': para_split[1], 'padding': para_split[2]}
        maxpool_list.append(maxpool_dict)
    return JsonResponse({"params": maxpool_list})


@csrf_exempt
def maxpooling_handling(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    pool_id = int(request.POST['pool_id'])
    strides = int(request.POST['strides'])
    padding = str(request.POST['padding']).strip()
    index_no = data_base.create_index_model(project_id)
    parameters = str(pool_id) + "," + str(strides) + "," + padding
    l_name = "maxpool"
    data_base.image_process_model_func(project_id, str(index_no), l_name, parameters)
    res_df = spark.sql(
        "SELECT * FROM image_process_model_table where project_id={0} order by index_no+0 ASC".format(project_id))
    res_df.show()
    res_df = res_df.drop('project_id')
    index_no = []
    layer_name = []
    parameters = []
    for each in res_df.select('index_no').collect():
        index_no.append(each.index_no)
    for each in res_df.select('layer_name').collect():
        layer_name.append(each.layer_name)
    for each in res_df.select('parameters').collect():
        parameters.append(each.parameters)
    img_path = maxpooling_function(project_id, index_no, layer_name, parameters)
    img_path = str(project_id) + "/" + img_path
    dataframe = data_base.show_list(project_id, l_name)
    dataframe.show()
    dataframe.drop('project_id')
    parameters = []
    maxpool_list = []
    maxpool_dict = {}
    index_no = []
    for each in dataframe.select('index_no').collect():
        index_no.append(each.index_no)
    for each in dataframe.select('parameters').collect():
        parameters.append(each.parameters)
    for i in range(len(parameters)):
        para_split = parameters[i].split(',')
        maxpool_dict = {'index_no': index_no[i], 'pool_size': para_split[0], 'strides': para_split[1],
                        'padding': para_split[2]}
        maxpool_list.append(maxpool_dict)
    context = {'img_name': img_path}
    html = render_to_string('ImageProcessing/filters-display.html', request=request, context=context)
    return JsonResponse({'I4': html, "params": maxpool_list})


@csrf_exempt
def maxpooling_delete(request):
    project_id = int(request.POST['project_id'])
    layer_no = int(request.POST['layer_no'])
    spark.sql("INSERT OVERWRITE TABLE image_process_model_table SELECT * FROM image_process_model_table "
              "WHERE project_id!={0} or index_no!='{1}'".format(project_id, layer_no))
    return JsonResponse({'success': "success"})


@csrf_exempt
def dropout_show_list(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    layer_name = "dropout"
    dataframe = data_base.show_list(project_id, layer_name)
    dataframe.show()
    dataframe.drop('project_id')
    parameters = []
    dropout_list = []
    dropout_dict = {}
    index_no = []
    for each in dataframe.select('index_no').collect():
        index_no.append(each.index_no)
    for each in dataframe.select('parameters').collect():
        parameters.append(each.parameters)
    for i in range(len(parameters)):
        para_split = parameters[i].split(',')
        dropout_dict = {'index_no': index_no[i], 'rate': para_split[0], 'seed': para_split[1]}
        dropout_list.append(dropout_dict)
    return JsonResponse({"params": dropout_list})


@csrf_exempt
def dropout_handling(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    rate = float(request.POST['rate'])
    seed = int(request.POST['seed'])
    index_no = data_base.create_index_model(project_id)
    parameters = str(rate) + "," + str(seed)
    l_name = "dropout"
    data_base.image_process_model_func(project_id, str(index_no), l_name, parameters)
    res_df = spark.sql(
        "SELECT * FROM image_process_model_table where project_id={0} order by index_no+0 ASC".format(project_id))
    res_df.show()
    res_df = res_df.drop('project_id')
    index_no = []
    layer_name = []
    parameters = []
    for each in res_df.select('index_no').collect():
        index_no.append(each.index_no)
    for each in res_df.select('layer_name').collect():
        layer_name.append(each.layer_name)
    for each in res_df.select('parameters').collect():
        parameters.append(each.parameters)
    img_path = dropout_function(project_id, index_no, layer_name, parameters)
    img_path = str(project_id) + "/" + img_path
    dataframe = data_base.show_list(project_id, l_name)
    dataframe.show()
    dataframe.drop('project_id')
    parameters = []
    dropout_list = []
    dropout_dict = {}
    index_no = []
    for each in dataframe.select('index_no').collect():
        index_no.append(each.index_no)
    for each in dataframe.select('parameters').collect():
        parameters.append(each.parameters)
    for i in range(len(parameters)):
        para_split = parameters[i].split(',')
        dropout_dict = {'index_no': index_no[i], 'rate': para_split[0], 'seed': para_split[1]}
        dropout_list.append(dropout_dict)
    context = {'img_name': img_path}
    html = render_to_string('ImageProcessing/filters-display.html', request=request, context=context)
    return JsonResponse({'I5': html, "params": dropout_list})


@csrf_exempt
def dropout_delete(request):
    project_id = int(request.POST['project_id'])
    layer_no = int(request.POST['layer_no'])
    spark.sql("INSERT OVERWRITE TABLE image_process_model_table SELECT * FROM image_process_model_table "
                  "WHERE project_id!={0} or index_no!='{1}'".format(project_id, layer_no))
    return JsonResponse({'success': "success"})


@csrf_exempt
def flatten_handling(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    params = str(0)
    l_name = "flatten"
    data_base.image_process_fit_func(project_id, l_name, params)
    return JsonResponse({'success': "success"})


@csrf_exempt
def dense_show_list(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    dataframe = data_base.dense_table_show_list(project_id)
    dataframe.show()
    dataframe.drop('project_id')
    units = []
    activation = []
    dense_list = []
    dense_dict = {}
    index_no = []
    for each in dataframe.select('index_no').collect():
        index_no.append(each.index_no)
    for each in dataframe.select('units').collect():
        units.append(each.units)
    for each in dataframe.select('activation').collect():
        activation.append(each.activation)
    for i in range(len(index_no)):
        dense_dict = {'index_no': index_no[i], 'units': units[i], 'activation': activation[i]}
        dense_list.append(dense_dict)
    return JsonResponse({"params": dense_list})


@csrf_exempt
def dense_handling(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    unit = int(request.POST['unit'])
    activation = str(request.POST['activation']).strip()
    index_no = data_base.create_index_dense(project_id)
    data_base.image_process_dense_func(project_id, str(index_no), str(unit), activation)
    dataframe = data_base.dense_table_show_list(project_id)
    dataframe.show()
    dataframe.drop('project_id')
    units = []
    activation = []
    dense_list = []
    dense_dict = {}
    index_no = []
    for each in dataframe.select('index_no').collect():
        index_no.append(each.index_no)
    for each in dataframe.select('units').collect():
        units.append(each.units)
    for each in dataframe.select('activation').collect():
        activation.append(each.activation)
    for i in range(len(index_no)):
        dense_dict = {'index_no': index_no[i], 'units': units[i], 'activation': activation[i]}
        dense_list.append(dense_dict)
    return JsonResponse({'success': "success", 'params':dense_list})


@csrf_exempt
def dense_delete(request):
    project_id = int(request.POST['project_id'])
    layer_no = int(request.POST['layer_no'])
    spark.sql("INSERT OVERWRITE TABLE image_process_dense_table SELECT * FROM image_process_dense_table "
              "WHERE project_id!={0} or index_no!='{1}'".format(project_id, layer_no))
    return JsonResponse({'success': "success"})


@csrf_exempt
def fit_show_list(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    layer_name = "fit"
    dataframe = data_base.fit_table_show_list(project_id, layer_name)
    dataframe.show()
    dataframe.drop('project_id')
    parameters = []
    fit_list = []
    fit_dict = {}
    for each in dataframe.select('params').collect():
        parameters.append(each.params)
    for i in range(len(parameters)):
        para_split = parameters[i].split(',')
        fit_dict = {'batch_size': para_split[0], 'epochs': para_split[1], 'verbose': para_split[2], 'val_split': para_split[3]}
        fit_list.append(fit_dict)
    return JsonResponse({"params": fit_list})


@csrf_exempt
def fit_handling(request):
    project_id = int(request.POST['project_id'])
    data_base = SparkDataBase()
    project_name = data_base.retreive_project_name(project_id)
    batch_id = int(request.POST['batch_id'])
    epochs = int(request.POST['epochs'])
    verbose = int(request.POST['verbose'])
    val = float(request.POST['val'])
    params = str(batch_id) + "," + str(epochs) + "," + str(verbose) + "," + str(val)
    l_name = "fit"
    data_base.image_process_fit_func(project_id, l_name, params)
    model_df = spark.sql(
        "SELECT * FROM image_process_model_table where project_id={0} order by index_no+0 ASC".format(project_id))
    model_df = model_df.drop('project_id')
    index_no = []
    layer_name = []
    parameters = []
    for each in model_df.select('index_no').collect():
        index_no.append(each.index_no)
    for each in model_df.select('layer_name').collect():
        layer_name.append(each.layer_name)
    for each in model_df.select('parameters').collect():
        parameters.append(each.parameters)

    dense_df = spark.sql(
        "SELECT * FROM image_process_dense_table where project_id={0} order by index_no+0 ASC".format(project_id))
    dense_df = dense_df.drop('project_id')
    units = []
    activation = []
    for each in dense_df.select('units').collect():
        units.append(each.units)
    for each in dense_df.select('activation').collect():
        activation.append(each.activation)

    fit_df = spark.sql(
        "SELECT * FROM image_process_fit_table where project_id={0}".format(project_id))
    fit_df = fit_df.drop('project_id')
    l_name = []
    params = []
    for each in fit_df.select('l_name').collect():
        l_name.append(each.l_name)
    for each in fit_df.select('params').collect():
        params.append(each.params)
    a = fit_function(project_id, index_no, layer_name, parameters, l_name, params, units, activation)
    return JsonResponse({'success': "success", 'a': a})

@csrf_exempt
def validate_handling(request):
    project_id = int(request.POST['project_id'])
    html = render_to_string('ImageProcessing/validation.html', request=request)
    return JsonResponse({'I6': html})

@csrf_exempt
def image_handling(request):
    project_id = int(request.POST['project_id'])
    img_name = request.FILES['img']
    tmp_file = os.path.join(settings.UPLOAD_PATH, img_name.name)
    path = default_storage.save(tmp_file, ContentFile(img_name.read()))
    img_url = os.path.join(settings.MEDIA_URL, path)
    prediction = predict_cell(project_id, img_url)
    return JsonResponse({'predict': prediction})


@csrf_exempt
def cf_basicstats(request):
    project_id = int(request.POST['project_id'])
    be_name = request.POST['be_name']
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(int(project_id))

    subset_df = data_frame[data_frame['billing_entity'] == be_name]

    if be_name == 'All Adventist W':
        daily_stats = daily_transformation(subset_df)["total_payments"].describe()
        daily_stats = daily_stats.reset_index()
        daily_stats.columns = ['Index', 'Daily_stats']

        weekly_stats = weekly_transformation(subset_df)["total_payments"].describe()
        weekly_stats = weekly_stats.reset_index()
        weekly_stats.columns = ['Index', 'Weekly_stats']

        monthly_stats = monthly_transformation(subset_df)["total_payments"].describe().reset_index()
        monthly_stats.columns = ['Index', 'Monthly_stats']

        basicStats = pd.merge(daily_stats, weekly_stats, on='Index')
        basicStats = pd.merge(basicStats, monthly_stats, on='Index')

        basicStats.set_index('Index', inplace=True)
        basicStats.drop('count', axis=0, inplace=True)

        basicStats_html = basicStats.to_html(classes=["table", "table-responsive", "table-bordered", "table-hover"])
        basic_stats_render = render_to_string('Cash_Forecasting/basic_stats.html', request=request)
        return JsonResponse({'basicStats': basicStats_html, 'basic_stats_html': basic_stats_render})

    else:
        daily_stats = daily_transformation(subset_df)["total_payments"].describe()
        daily_stats = daily_stats.reset_index()
        daily_stats.columns = ['Index', 'Daily_stats']

        weekly_stats = weekly_transformation(subset_df)["total_payments"].describe()
        weekly_stats = weekly_stats.reset_index()
        weekly_stats.columns = ['Index', 'Weekly_stats']

        monthly_stats = monthly_transformation(subset_df)["total_payments"].describe().reset_index()
        monthly_stats.columns = ['Index', 'Monthly_stats']

        basicStats = pd.merge(daily_stats, weekly_stats, on='Index')
        basicStats = pd.merge(basicStats, monthly_stats, on='Index')

        basicStats.set_index('Index', inplace=True)
        basicStats.drop('count', axis=0, inplace=True)

        no_of_months = str(count_of_months(subset_df))

        df = pd.DataFrame()
        df = data_frame.groupby(['billing_entity', 'MonthYear'])['total_payments'].sum().reset_index()
        df = pd.DataFrame(df.groupby('billing_entity')['total_payments'].mean())  # avg of monthly payments of each BE
        df.rename(columns={'total_payments': 'monthly_mean'}, inplace=True)
        df.drop('All Adventist W', axis=0, inplace=True)

        labels = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        df1 = pd.DataFrame(df.describe())
        df1.columns = ['monthly_mean']
        BE_monthly_res = dict(zip(labels, df1['monthly_mean']))

        ############## Means of a particular Billing Entity ########################

        subset_monthly_mean = int(monthly_stats[monthly_stats.Index == "mean"]['Monthly_stats'])

        ###########################TOTAL BE COMPARISON CHECK#########################

        total_comp_monthly = [BE_monthly_res["75%"], BE_monthly_res["50%"], BE_monthly_res["25%"]]

        category = ""
        if subset_monthly_mean >= total_comp_monthly[0]:
            category = "LARGE"
        elif total_comp_monthly[0] > subset_monthly_mean >= total_comp_monthly[1]:
            category = "MEDIUM"
        else:
            category = "SMALL"

        ############### Highlighting the selected be mean in the boxplot#################################
        be_monthly_mean = round(float(monthly_stats[monthly_stats["Index"] == "mean"]['Monthly_stats']), 2)
        all_be_mean_list = list(df['monthly_mean'].apply(lambda x: round(x, 2)))
        be_mean_index = all_be_mean_list.index(be_monthly_mean)
        monthly_box_plot = payments_box_plot(df['monthly_mean'], be_mean_index)

        basicStats_html = basicStats.to_html(classes=["table", "table-responsive", "table-bordered", "table-hover"])
        basic_stats_render = render_to_string('Cash_Forecasting/basic_stats.html', request=request)
        return JsonResponse({'basicStats': basicStats_html, 'basic_stats_html': basic_stats_render,
                             'months_count': no_of_months, 'category': category, 'monthly_box_plot': monthly_box_plot})


@csrf_exempt
def cf_timeseries(request):
    project_id = int(request.POST['project_id'])
    be_name = request.POST['be_name']
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(int(project_id))

    # subset_df = subset_be_data(be_name, data_frame)
    subset_df = data_frame[data_frame['billing_entity'] == be_name]

    monthly_sum = monthly_transformation(subset_df)
    rolling_plot_html = timeseries_scatter_plot(monthly_sum)
    # line_chart_html = timeseries_line_chart_plot(monthly_sum2, years1, sampling_input)
    line_chart_html = timeseries_line_chart_plot(monthly_sum)
    bar_plot_html = timeseries_bar_plot(monthly_sum)

    time_series_render = render_to_string('Cash_Forecasting/time_series_plots.html', request=request)
    return JsonResponse({'ts_html': time_series_render, 'rolling_plot_html': rolling_plot_html,
                         'line_chart_html': line_chart_html, 'bar_plot_html': bar_plot_html})


@csrf_exempt
def cf_weekdays(request):
    project_id = int(request.POST['project_id'])
    be_name = request.POST['be_name']
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(int(project_id))

    # subset_df = subset_be_data(be_name, data_frame)
    subset_df = data_frame[data_frame['billing_entity'] == be_name]
    weekday_sum_merged, monthyears, weekday_sum, weekday_sum_mean = weekday_transformation(subset_df)

    wd_bar_plot_html = wd_bar_plot(weekday_sum_mean)
    wd_line_chart_html = wd_line_chart(weekday_sum, monthyears)
    # wd_decompose_graph_html = wd_decompose_graph(resample)

    weekdays_html_render = render_to_string('Cash_Forecasting/weekday_plots.html', request=request)
    return JsonResponse({'wd_html': weekdays_html_render, 'wd_bar_plot_html': wd_bar_plot_html,
                         'wd_line_chart_html': wd_line_chart_html})


@csrf_exempt
def cf_encounter_class(request):
    project_id = int(request.POST['project_id'])
    be_name = request.POST['be_name']
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(int(project_id))
    # subset_df = subset_be_data(be_name, data_frame)
    subset_df = data_frame[data_frame['billing_entity'] == be_name]

    encounter_class_pivot, enc_overall = encounter_transformation(subset_df)
    ec_line_chart_html = ec_line_plot(encounter_class_pivot)
    ec_bar_plot_html = ec_bar_plot(enc_overall)
    ec_footfall_html = ec_footfall_plot(subset_df)
    ec_charges_html = ec_charges_plot(subset_df)

    encounter_class_html_render = render_to_string('Cash_Forecasting/encounter_class.html', request=request)
    return JsonResponse({'ec_html': encounter_class_html_render, 'ec_line_chart_html': ec_line_chart_html,
                         'ec_bar_plot_html': ec_bar_plot_html, 'ec_footfall_html': ec_footfall_html,
                         'ec_charges_html': ec_charges_html})


@csrf_exempt
def cf_financial_class(request):
    project_id = int(request.POST['project_id'])
    be_name = request.POST['be_name']
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(int(project_id))
    # dic = create_dic(data_frame)
    # subset_df = subset_be_data(be_name, data_frame)
    subset_df = data_frame[data_frame['billing_entity'] == be_name]
    f_data_monthly, f_data_line_pivot = financial_transformation(subset_df)
    fc_line_plot_html = fc_line_plot(f_data_line_pivot)
    fc_bar_plot_html = fc_bar_plot(f_data_monthly)
    fc_footfall_html = fc_footfall(subset_df)
    fc_charges_html = fc_charges(subset_df)

    financial_class_html_render = render_to_string('Cash_Forecasting/financial_class.html', request=request)
    return JsonResponse({'fc_html': financial_class_html_render, 'fc_line_chart_html': fc_line_plot_html,
                         'fc_bar_plot_html': fc_bar_plot_html, 'fc_footfall_html': fc_footfall_html,
                         'fc_charges_html': fc_charges_html})


@csrf_exempt
def cf_xregs(request):
    xreg_html_render = render_to_string('Cash_Forecasting/xregs.html', request=request)
    return JsonResponse({'xr_html': xreg_html_render})


@csrf_exempt
def cf_xregs_select(request):
    project_id = request.GET['project_id']
    be_name = request.GET['be_name']
    xreg = request.GET['x_regs']
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(int(project_id))
    # dic = create_dic(data_frame)
    # subset_df = subset_be_data(be_name, data_frame)
    subset_df = data_frame[data_frame['billing_entity'] == be_name]
    monthly_res = monthly_transformation(subset_df)
    monthly_res_xregs = monthly_transformation_ext_reg(subset_df, xreg)
    ts_ext_reg_plot_html = timeseries_ext_reg(subset_df, monthly_res_xregs, xreg)
    lagged_plot_html = ff_lagged_plot(subset_df, monthly_res, xreg)
    exreg_correalation_plot_html = ff_per_month(subset_df, monthly_res, xreg)
    exreg_correalation_stats_html = ff_xreg_correlation(subset_df, monthly_res, xreg)
    exreg_correalation_stats_html = exreg_correalation_stats_html.to_html(classes=["table", "table-responsive", "table-bordered", "table-hover"])
    xreg_html_render = render_to_string('Cash_Forecasting/xregs.html', request=request)
    return JsonResponse({'xreg_plot_html_render': xreg_html_render,'ts_ext_reg_plot_html':ts_ext_reg_plot_html, 'lagged_plot_html': lagged_plot_html,
                         'exreg_correalation_plot_html': exreg_correalation_plot_html,
                         'exreg_correalation_stats_html': exreg_correalation_stats_html})


@csrf_exempt
def cf_acf_pacf(request):
    project_id = int(request.POST['project_id'])
    be_name = request.POST['be_name']
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(int(project_id))

    # subset_df = subset_be_data(be_name, data_frame)
    subset_df = data_frame[data_frame['billing_entity'] == be_name]

    monthly_data = monthly_transformation(subset_df)
    monthly_data['MonthYear'] = pd.to_datetime(monthly_data['MonthYear'])
    acf_df = monthly_data.iloc[:, 0:4]
    acf_df.set_index('MonthYear', inplace=True)
    corr_df = pd.DataFrame()
    # fetch ACF
    ac, confint_ac = (acf(acf_df['total_payments'], nlags=25, alpha=0.05))
    pac, confint_pac = (pacf(acf_df['total_payments'], nlags=25, alpha=0.05))
    corr_df['ACF'] = list(ac)
    corr_df['lower_acf'] = list(confint_ac[:, 0])
    corr_df['upper_acf'] = list(confint_ac[:, 1])
    corr_df['PACF'] = list(pac)
    corr_df['lower_pacf'] = list(confint_pac[:, 0])
    corr_df['upper_pacf'] = list(confint_pac[:, 1])
    acf_plot_html = acf_plot(corr_df)
    pacf_plot_html = pacf_plot(corr_df)
    acf_pacf_html_render = render_to_string('Cash_Forecasting/acf_pacf.html', request=request)
    return JsonResponse({'acf_pacf_html': acf_pacf_html_render, 'acf_plot_html': acf_plot_html,
                         'pacf_plot_html': pacf_plot_html})


@csrf_exempt
def cf_forecast(request):
    project_id = int(request.POST['project_id'])
    be_name = request.POST['be_name']
    data_base = SparkDataBase()
    data_frame = data_base.find_data_frame(int(project_id))
    subset_df = data_frame[data_frame['billing_entity'] == be_name]
    directory = r"mlp/SaveCashData/"

    n_periods = get_best_lag(subset_df, be_name)
    lags = n_periods

    monthly_data = monthly_transformation(subset_df)
    var = 'total_payments'
    series = monthly_data[var]
    series.index = pd.DatetimeIndex(monthly_data.MonthYear)

    ### Fetch external regressors- weekdays, Payor wise PMI, Payor wise footfall,Payor wise charges
    combined_wday, xreg_wday, pred_xreg_wday = wday_per_month_new(subset_df, 3)  # get weekdays
    percent_flag = 1  # convert PMI, charges and footfall data into percentage so that data is in same scale
    act_xreg_pmi, pred_xreg_pmi = ec_PMI(subset_df, 3, lags, percent_flag)  # get PMI
    f_agg_var = 'footfall'
    act_xreg_wd, pred_xreg_wd = ec_Xreg(subset_df, f_agg_var, 3, lags, percent_flag)  # get footfall
    c_agg_var = 'total_charge'
    act_xreg_cc, pred_xreg_cc = ec_Xreg(subset_df, c_agg_var, 3, lags, percent_flag)  # get charges

    ### Combine all external regressors
    df1 = pd.merge(act_xreg_pmi, act_xreg_wd, left_index=True, right_index=True)  # merge pmi with footfall
    df2 = pd.merge(df1, act_xreg_cc, left_index=True, right_index=True)  # merge df1 with charges
    act_xreg = df2.copy()  # copy df2
    ## Feature selection is done to remove less significant regressors
    avg_xreg = act_xreg.apply(lambda row: np.mean(row)).sort_values(ascending=False)  # check mean of df2 columns
    xreg_cols_varmax = avg_xreg[0:2].index
    xreg_all = act_xreg.apply(lambda row: np.std(row)).sort_values(ascending=False)  # check std of df2 columns
    xreg_com = pd.concat([avg_xreg, xreg_all], axis=1, sort=True)  # combine mean and sd
    xreg_com.columns = ['avg', 'std']
    xreg_cols = xreg_com[(xreg_com['avg'] > 5) & (xreg_com['std'] > 0.5)].index  # Feature Selection based on average percent contribution and variance
    df2 = df2[xreg_cols]  # filter out selected features from df2
    df2 = pd.concat([df2, xreg_wday], axis=1)  # combine weekdays data with df2
    df = pd.merge(pd.DataFrame(series), df2, on=df2.index, how='inner')  # final regressor dataframe to be used with models
    df.rename(columns={'key_0': 'MonthYear'}, inplace=True)
    df.set_index('MonthYear', inplace=True)
    df.fillna(0, inplace=True)

    ### similarly merge unseen data and filter out significant columns. This is required for forecast plots
    df_unseen = pd.merge(pred_xreg_pmi, pred_xreg_wd, left_index=True,
                         right_index=True)  # merge unseen pmi with footfall
    df_unseen = pd.merge(df_unseen, pred_xreg_cc, left_index=True,
                         right_index=True)  # merge unseen charges with above dataframe
    df_unseen = pd.concat([df_unseen, pred_xreg_wday], axis=1)  # merge weekdays of unseen months
    df_unseen['total_payments'] = np.nan  # fill NA payments for unseen payments as this format is compatible with the
    df_unseen = df_unseen[df.columns]  # reorder the columns just like df(i.e seen data)

    print("DF :",df.tail(5))
    print('DF_unseen',df_unseen)

    ################## START MODELLING ###############
    date = max(df.index) - relativedelta(months=3)
    data = df[:date]
    rem_data = 3
    varmax_cols = np.append(xreg_cols_varmax, xreg_wday.columns)
    dataV = df[:date][np.append(var, varmax_cols)]

    # For Arima
    # Divide train data into three sets:
    # 1. Train
    # 2. Arima xreg choosing set
    # 3. Test Set for Running all models and choosing best model

    train, test = train_test_split(data, n_test=3)
    trainV, testV = train_test_split(dataV, n_test=3)

    # train and test for non linear model- Random Forest
    x_train = train.iloc[:, 1:]
    y_train = train.iloc[:, 0]

    print("x_train:",x_train.tail(5))
    print("y_train:",y_train.tail(5))

    dt1 = date - relativedelta(months=rem_data)
    x_test, var_flag, var_mape = create_test(df, train, dt1, n_periods, rem_data)
    y_test = df.loc[x_test.index][var]

    print("x_test (predicted)",x_test)
    print("y_test",y_test)

    rf_model, fitted_rf, fimp_sorted, confint = random_forest(x_train, y_train, x_test)

    print("Random Forest Test predictions \n",fitted_rf)

    # ARIMA model
    best_model_arima, model_arima, fitted_arima, confint_arima = get_arima_best(x_train, y_train, y_test, x_test, var, rem_data)

    print("ARIMA Model Test predictions \n", fitted_arima)

    ########## Varmax Model ############
    model_varmax, fitted_varmax, confint = varmax_model_test(trainV, x_test, rem_data)

    print("VARMAX Model Test predictions \n",fitted_varmax)

    model_avg, fitted_avg, confint_avg = moving_average(y_train, x_test, rem_data)

    print("MA Model Test predictions \n", fitted_avg)

    # sort models based on least mape
    model_list = []
    # Create MAPE dictionary on test set
    model_list.append(['rf', mape(y_test, fitted_rf), rf_model])
    model_list.append(['varmax', mape(y_test, fitted_varmax), model_varmax])
    model_list.append(['moving_average', mape(y_test, fitted_avg), model_avg])
    model_list.append([best_model_arima, mape(y_test, fitted_arima), model_arima])
    model_list.sort(key=lambda tup: tup[1])

    print("Model List \n",model_list)

    ### Create validation set and check selected Model Performance
    x_val, var_flag, var_mape = create_test(df, data, date, n_periods, rem_data)
    y_val = df.loc[x_val.index][var]

    # recreate train
    x_train1 = data.iloc[:, 1:]
    y_train1 = data.iloc[:, 0]

    # model for seen data
    fit, fitted, best_model_name, test_confint = best_fit(x_train1, y_train1, x_val, y_val, model_list, rem_data, var, dataV)

    if best_model_name == 'rf':
        y_unseen, un_confint, plot_fc_test_train_html, test_mape = make_plot(be_name, var, y_train1, y_val, df_unseen,
                                                                                               fitted, df,
                                                                                               n_periods, rem_data, date, best_model_name,
                                                                                               model_list, dataV, test_confint)
        feature_imp_graph = rf_feature_importance(fit, x_train1)
        forecast_model_html_render = render_to_string('Cash_Forecasting/forecast_rf.html', request=request)
        return JsonResponse({'fm_html': forecast_model_html_render, 'fm_model_html': plot_fc_test_train_html,
                             'mape': test_mape, 'best_model': best_model_name, "rf_feature_imp_html": feature_imp_graph})
    else:
        y_unseen, un_confint, plot_fc_test_train_html, test_mape = make_plot(be_name, var, y_train1,y_val, df_unseen,
                                                                             fitted, df, n_periods, rem_data,
                                                                             date, best_model_name, model_list, dataV,
                                                                             test_confint)
    # make plot
        forecast_model_html_render = render_to_string('Cash_Forecasting/forecast_ar.html', request=request)
        return JsonResponse({'fm_html': forecast_model_html_render, 'fm_model_html': plot_fc_test_train_html,
                             'mape': test_mape, 'best_model': best_model_name})
