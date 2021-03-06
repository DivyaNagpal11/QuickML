from django.urls import path
from . import views

urlpatterns = [
    path('', views.main),
    path('dpm/', views.project_names),
    path('project/<int:value>/', views.project, name='project'),
    path('cps/', views.create_project, name='createproject'),
    path('getSidebar/', views.getSidebar),
    path('right_side_bar/', views.right_side_bar),
    path('summary/', views.summary, name='summary'),
    path('plotting/', views.plotting),
    path('distributionchart/', views.distribution_plot),
    path('scatterplot/', views.scatter_plot),
    path('boxplot/', views.box_plot),
    path('missingvalues/', views.missing_values),
    path('missingvalueimputation/', views.missing_value_impuation),
    path('normalization/', views.normalization),
    path('normalizationmethods/', views.normalization_methods),
    path('binning/', views.binning), #redirects to Binning Html and calls binning method from views.py
    path('binninghandler/', views.binning_handler), #calls binning_handler from views.py
    path('viewAllData/',views.view_All_Data),
    path('deletefeature/', views.deletefeature),
    path('deletefeatureshandling/', views.delete_features_handling),
    path('outliers/', views.outliers),
    path('outliers_handling/', views.outliers_handling),
    path('boxplot_outliers/', views.box_plot_outliers),
    path('distribution_plot_normalization/', views.distribution_plot_normalization),
    path('encoding/', views.encoding),
    path('encoding_method/', views.encode_method),
    path('sampling/', views.sampling),
    path('sampling_info/', views.sampling_info),
    path('sampling_methods/', views.sampling_methods),
    path('modelling/', views.modelling),
    path('get_algorithms/',views.type_of_algorithms),
    path('get_algorithm_parameters/',views.selected_algorithm_input_parameters),
    path('edit_algorithm_parameters/',views.edit_algorithm_input_parameters),
    path('algorithms/', views.algorithms),
    path('undo/', views.current_data_frame_one_step_back),
    path('validation/',views.validation),
    path('validations_data/',views.validation_data),
    path('prediction/',views.predict_output),
    path('save_model_for_api/',views.save_model_for_api),
    path('save_algorithm_input_parameters/',views.save_input_parameters),
    path('clustering/', views.clustering),
    path('get_cluster_algorithm_parameters/', views.selected_cluster_algorithm_input_parameters),
    path('edit_cluster_algorithm_parameters/', views.edit_cluster_algorithm_input_parameters),
    path('save_cluster_algorithm_input_parameters/',views.save_cluster_input_parameters),
    path('cluster_algorithms/', views.cluster_algorithms),
    path('cluster_validation/', views.cluster_validation),
    path('cluster_validation_data/', views.cluster_validation_data),
    path('cluster_prediction/',views.cluster_predict_output),
    path('cluster_save_model_for_api/', views.save_cluster_model_for_api),
    path('normalPlot/', views.plot_normal_distribution),
    path('normalDistributionchart/', views.normal_distribution),
    path('normalDistributionFunc/', views.distribute_normal),
    path('regression/', views.Regression_load),
    path('regression_algorithms/', views.Regression_algorithms),
    path('regression_api_save/', views.Regression_model_save_api),
    path('regression_validation/', views.regression_validation),
    path('regression_validation_data/', views.regression_validation_data),
    path('regression_prediction/', views.regression_predict_output),
    path('Deletion/', views.Delete_proj),
    path('nn_visual/', views.visual_handling),
    path('nn_model/', views.model_handling),
    path('nn_convolution/', views.convolution_show_list),
    path('nn_model_conv/', views.conv2d_handling),
    path('nn_model_del_conv/', views.conv2d_delete),
    path('nn_maxpool/', views.maxpool_show_list),
    path('nn_model_max_pool/', views.maxpooling_handling),
    path('nn_model_del_maxpool/', views.maxpooling_delete),
    path('nn_dropout/', views.dropout_show_list),
    path('nn_model_drop/', views.dropout_handling),
    path('nn_model_del_dropout/', views.dropout_delete),
    path('nn_model_flatten/', views.flatten_handling),
    path('nn_model_dense/', views.dense_handling),
    path('nn_dense/', views.dense_show_list),
    path('nn_model_del_dense/', views.dense_delete),
    path('nn_fit/', views.fit_show_list),
    path('nn_model_fit/', views.fit_handling),
    path('nn_validate/', views.validate_handling, name='nn_validate'),
    path('nn_validate_image/', views.image_handling),
    path('cf_basicstats/', views.cf_basicstats),
    path('cf_timeseries/', views.cf_timeseries),
    path('cf_weekdays/', views.cf_weekdays),
    path('cf_encounter_class/', views.cf_encounter_class),
    path('cf_financial_class/', views.cf_financial_class),
    path('cf_xregs/', views.cf_xregs),
    path('cf_xregs_select/', views.cf_xregs_select),
    path('cf_acf_pacf/', views.cf_acf_pacf),
    path('cf_forecast/', views.cf_forecast),
]
