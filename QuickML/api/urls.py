from django.urls import path
from . import views

urlpatterns = [
    path('<str:model_id>/', views.predict_api),
    path('clustering/<str:model_id>/', views.cluster_predict_api),
    path('regression/<str:model_id>/', views.regression_predict_api),
]

