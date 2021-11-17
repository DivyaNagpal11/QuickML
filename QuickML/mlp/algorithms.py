import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from .spark_database import *

class Algorithms:
    def __init__(self, data_frame, project_id):
        self.data_frame = data_frame.dropna()
        self.project_id = project_id

    def data_processing(self, data_options, algorithm_info):
        algorithm = algorithm_info[1]
        data_base = SparkDataBase()
        input_parameters = data_base.load_models_input_parameters(self.project_id, algorithm)
        features_dtypes = self.data_frame.dtypes.apply(lambda feature: feature.name).to_dict()
        target_feature = data_options[0]
        x, y = self.target_feature_split(target_feature)
        input_features = list(x)
        test_size = data_options[1]
        if test_size < 1:
            test_size = float(test_size)
        else:
            test_size = int(test_size)
        x_train, x_test, y_train, y_test = self.data_train_test_split(x, y, test_size, int(data_options[2]))
        if data_options[3] == 1:
            x_train, x_test = self.scaling(x_train, x_test)
        result_data = []
        result_data.append(int(algorithm_info[0]))
        if algorithm == "K Nearest Neighbour":
            accuracy_data = self.KNN_algorithm(x_train, x_test, y_train, y_test,input_features,target_feature,
                                           features_dtypes,input_parameters)
            result_data.append(accuracy_data)
        elif algorithm == "Decision Tree":
            accuracy_data = self.decision_tree(x_train, x_test, y_train, y_test,input_features,target_feature,
                                           features_dtypes,input_parameters)
            result_data.append(accuracy_data)
        elif algorithm == "Random Forest":
            accuracy_data = self.random_forest(x_train, x_test, y_train, y_test,input_features,target_feature,
                                           features_dtypes,input_parameters)
            result_data.append(accuracy_data)
        elif algorithm == "Support Vector Machine":
            accuracy_data = self.SVM(x_train, x_test, y_train, y_test,input_features,target_feature,features_dtypes,
                                    input_parameters)
            result_data.append(accuracy_data)
        else:
            accuracy_data = self.XgBoost(x_train, x_test, y_train, y_test,input_features,target_feature,features_dtypes,
                                     input_parameters)
            result_data.append(accuracy_data)
        return result_data

    def target_feature_split(self, target):
        x = self.data_frame.drop(target, axis=1)
        y = self.data_frame[target]
        return x, y

    def data_train_test_split(self, x, y, test_size, random_state_value):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state_value)
        return x_train, x_test, y_train, y_test

    def scaling(self, x_train, x_test):
        sc = StandardScaler()
        sc.fit(x_train)
        x_train = sc.transform(x_train)
        x_test = sc.transform(x_test)
        return x_train, x_test

    def KNN_algorithm(self, x_train, x_test, y_train, y_test,input_features,target_feature,features_dtypes,input_parameters):
        p=input_parameters[4]
        if p=="euclidean_distance":
            p=2
        else:
            p=1
        knn = KNeighborsClassifier(n_neighbors=int(input_parameters[0]),weights=input_parameters[1],
                                   algorithm=input_parameters[2], leaf_size=int(input_parameters[3]),
                                   p=p, metric='minkowski')
        knn.fit(x_train, y_train)
        data_base = SparkDataBase()
        pickle_model = pickle.dumps(knn)
        data = []
        train_accuracy = (knn.score(x_train, y_train))
        data.append(round(train_accuracy, 2))
        test_accuracy = (knn.score(x_test, y_test))
        data.append(round(test_accuracy, 2))
        y_pred = knn.predict(x_test)
        overall_accuracy = accuracy_score(y_test, y_pred)
        data.append(round(overall_accuracy, 2))
        data_base.save_model(self.project_id, "K Nearest Neighbour",round(train_accuracy,2),round(test_accuracy,2),round(overall_accuracy,2),target_feature,knn)
        return data

    def decision_tree(self, x_train, x_test, y_train, y_test,input_features,target_feature,features_dtypes,input_parameters):
        max_depth=int(input_parameters[2])
        if max_depth == 0:
            max_depth=None
        try:
            min_samples_split = int(input_parameters[3])
        except ValueError:
            min_samples_split = float(input_parameters[3])
        try:
            min_samples_leaf= int(input_parameters[4])
        except ValueError:
            min_samples_leaf=float(input_parameters[4])
        max_features = input_parameters[6]
        if max_features=="None":
            max_features=None
        random_state=int(input_parameters[7])
        if random_state == 0:
            random_state = None
        max_leaf_nodes=int(input_parameters[8])
        if max_leaf_nodes==0:
            max_leaf_nodes=None
        presort=input_parameters[10]
        if presort == "true":
            presort = True
        else:
            presort = False
        decision_tree = tree.DecisionTreeClassifier(criterion=input_parameters[0],splitter=input_parameters[1],
                                                    max_depth=max_depth,min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=float(input_parameters[5]),
                                                    max_features=max_features,random_state=random_state,
                                                    max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=float(input_parameters[9]),
                                                    presort=presort
                                                    )
        decision_tree.fit(x_train, y_train)
        pickle_model = pickle.dumps(decision_tree)
        data_base = SparkDataBase()
        data = []
        train_accuracy = (decision_tree.score(x_train, y_train))
        data.append(round(train_accuracy, 2))
        test_accuracy = (decision_tree.score(x_test, y_test))
        data.append(round(test_accuracy, 2))
        y_pred = decision_tree.predict(x_test)
        overall_accuracy = accuracy_score(y_test, y_pred)
        data.append(round(overall_accuracy, 2))
        data_base.save_model(self.project_id, "Decision Tree",round(train_accuracy,2),round(test_accuracy,2),round(overall_accuracy,2),target_feature,decision_tree)
        return data

    def random_forest(self, x_train, x_test, y_train, y_test,input_features,target_feature,features_dtypes,input_parameters):
        max_depth = int(input_parameters[2])
        if max_depth == 0:
            max_depth = None
        try:
            min_samples_split = int(input_parameters[3])
        except ValueError:
            min_samples_split = float(input_parameters[3])
        try:
            min_samples_leaf = int(input_parameters[4])
        except ValueError:
            min_samples_leaf = float(input_parameters[4])
        max_features = input_parameters[6]
        if max_features=="None":
            max_features=None
        max_leaf_nodes=int(input_parameters[7])
        if max_leaf_nodes==0:
            max_leaf_nodes=None
        bootstrap=input_parameters[9]
        if bootstrap=="true":
            bootstrap=True
        else:
            bootstrap=False
        obb_score = input_parameters[10]
        if obb_score == "true":
            obb_score = True
        else:
            obb_score = False
        random_state = int(input_parameters[11])
        if random_state==0:
            random_state=None
        warm_start=input_parameters[13]
        if warm_start=="true":
            warm_start=True
        else:
            warm_start=False
        random_forest = RandomForestClassifier(n_estimators=int(input_parameters[0]),criterion=input_parameters[1],
                                               max_depth=max_depth,min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=float(input_parameters[5]),
                                               max_features=max_features,max_leaf_nodes=max_leaf_nodes,
                                               min_impurity_decrease=float(input_parameters[8]), bootstrap=bootstrap,
                                               oob_score=obb_score,random_state=random_state,
                                               verbose=int(input_parameters[12]),warm_start=warm_start
                                               )
        random_forest.fit(x_train, y_train)
        pickle_model = pickle.dumps(random_forest)
        data_base = SparkDataBase()
        data = []
        train_accuracy = (random_forest.score(x_train, y_train))
        data.append(round(train_accuracy, 2))
        test_accuracy = (random_forest.score(x_test, y_test))
        data.append(round(test_accuracy, 2))
        y_pred = random_forest.predict(x_test)
        overall_accuracy = accuracy_score(y_test, y_pred)
        data.append(round(overall_accuracy, 2))
        data_base.save_model(self.project_id, "Random Forest",round(train_accuracy,2),round(test_accuracy,2),round(overall_accuracy,2),target_feature,random_forest)
        return data

    def SVM(self,x_train, x_test, y_train, y_test,input_features,target_feature,features_dtypes,input_parameters):
        gamma=float(input_parameters[3])
        if gamma==0.0:
            gamma="auto"
        shrinking=input_parameters[5]
        if shrinking=="true":
            shrinking = True
        else:
            shrinking = False
        probability = input_parameters[6]
        if probability == "true":
            probability = True
        else:
            probability = False
        verbose = input_parameters[8]
        if verbose == "true":
            verbose = True
        else:
            verbose = False
        random_state=int(input_parameters[11])
        if random_state == 0:
            random_state = None
        svm = SVC(C=float(input_parameters[0]),kernel=input_parameters[1],degree=int(input_parameters[2]),
                  gamma=gamma, coef0=float(input_parameters[4]),shrinking=shrinking,probability=probability,
                  tol=float(input_parameters[7]),verbose=verbose,max_iter=int(input_parameters[9]),
                  decision_function_shape=input_parameters[10],random_state=random_state
                  )
        svm.fit(x_train, y_train)
        pickle_model = pickle.dumps(svm)
        data_base = SparkDataBase()
        data = []
        train_accuracy=(svm.score(x_train, y_train))
        data.append(round(train_accuracy,2))
        test_accuracy = (svm.score(x_test, y_test))
        data.append(round(test_accuracy, 2))
        y_pred = svm.predict(x_test)
        overall_accuracy=accuracy_score(y_test, y_pred)
        data.append(round(overall_accuracy, 2))
        data_base.save_model(self.project_id, "Support Vector Machine",round(train_accuracy,2),round(test_accuracy,2),round(overall_accuracy,2),target_feature,svm)
        return data

    def XgBoost(self, x_train, x_test, y_train, y_test,input_features,target_feature,features_dtypes,input_parameters):
        data = []
        xgb_clf = xgb.XGBClassifier(max_depth=int(input_parameters[0]),learning_rate=float(input_parameters[1]),
                                    n_estimators=int(input_parameters[2]),booster=input_parameters[3],
                                    gamma=float(input_parameters[4]),min_child_weight=int(input_parameters[5]),
                                    max_delta_step=int(input_parameters[6]),subsample=int(input_parameters[7]),
                                    colsample_bytree=float(input_parameters[8]),colsample_bylevel=float(input_parameters[9]),
                                    reg_alpha=float(input_parameters[10]),reg_lambda=float(input_parameters[11]),
                                    scale_pos_weight=float(input_parameters[12]),base_score=float(input_parameters[13]),
                                    random_state=int(input_parameters[14])
                                    )
        xgb_clf = xgb_clf.fit(x_train, y_train)
        pickle_model = pickle.dumps(xgb_clf)
        data_base = SparkDataBase()
        train_accuracy = round(xgb_clf.score(x_train, y_train),2)
        data.append(train_accuracy)
        test_accuracy = round((xgb_clf.score(x_test, y_test)),2)
        data.append(test_accuracy)
        y_pred = xgb_clf.predict(x_test)
        accuracy = round(accuracy_score(y_test, y_pred),2)
        data.append(accuracy)
        data_base.save_model(self.project_id, "XgBoost",train_accuracy,test_accuracy,accuracy,target_feature,xgb_clf)
        return data


def execute_selected_model(model, test):
    predict = model.predict(test)
    return predict

