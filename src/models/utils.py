import importlib
import time
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, average_precision_score, balanced_accuracy_score, accuracy_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

def generate_model_config(config):
    if config['model_type']=='classification':
        models_sklearn = { 'XGBoost': 'xgboost.XGBClassifier',
                           'RandomForest': 'sklearn.ensemble.RandomForestClassifier',
                           'ExtraTrees': 'sklearn.ensemble.ExtraTreesClassifier',
                           'AdaBoost': 'sklearn.ensemble.AdaBoostClassifier',
                           'LogisticRegression': 'sklearn.linear_model.LogisticRegression',
                           'SVM': 'sklearn.svm.SVC',
                           'GradientBoostingClassifier': 'sklearn.ensemble.GradientBoostingClassifier',
                           'DecisionTreeClassifier': 'sklearn.tree.DecisionTreeClassifier',
                           'SGDClassifier': 'sklearn.linear_model.SGDClassifier',
                           'KNeighborsClassifier': 'sklearn.neighbors.KNeighborsClassifier'
                          }
    elif config['model_type']=='regression':
        models_sklearn = { 'XGBoost': 'xgboost.XGBRegressor',
                           'RandomForest': 'sklearn.ensemble.RandomForestRegressor',
                           'ExtraTrees': 'sklearn.ensemble.ExtraTreesRegressor',
                           'AdaBoost': 'sklearn.ensemble.AdaBoostRegressor',
                           'GradientBoostingRegressor': 'sklearn.ensemble.GradientBoostingRegressor',
                           'DecisionTreeRegressor': 'sklearn.tree.DecisionTreeRegressor',
                           'SGDClassifier': 'sklearn.linear_model.SGDRegressor',
                           'KNeighborsClassifier': 'sklearn.neighbors.KNeighborsRegressor'
                          }

    model_config = {}
    models = config['model_params']['model']
    for model in models:
        if 'class_weight' in config['model_params']['parameters'][model]:
            if 'None' in config['model_params']['parameters'][model]['class_weight']:
                new_list = config['model_params']['parameters'][model]['class_weight']
                new_list.remove('None')
                new_list.append(None)
                config['model_params']['parameters'][model]['class_weight'] = new_list
            
        model_config[models_sklearn[model]] = config['model_params']['parameters'][model]

    return model_config

def generate_gridsearch_configs(grid_config):
    """Flattens a model/parameter grid configuration into individually
    trainable model/parameter pairs
    Yields: (tuple) classpath and parameters
    """
    for class_path, parameter_config in grid_config.items():
        for parameters in ParameterGrid(parameter_config):
            yield class_path, parameters

def train_model(X_train, y_train, class_path, parameters, log):
    """Fit a model to a training set. Works on any modeling class that
    is available in this package's environment and implements .fit
    Args:
        class_path (string) A full classpath to the model class, i.e. 'sklearn.neighbors.KNeighborsClassifier'
        parameters (dict) hyperparameters to give to the model constructor
    Returns:
        tuple of (fitted model, list of column names without label)
    """
    start = time.time()
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    instance = cls(**parameters)
    log['model']=class_path
    log = {**parameters, **log}
    train_time = time.time() - start
    log['train_time']=train_time
        
    return instance.fit(X_train, y_train), log

def get_metrics(X, y, trained_model, log, pred_type, config):
    ''' Use model to predict on data, and get metrics for it's performance.
    Args:
        X: Features matrix that data will be the input to the model
        y: ground truth for features matrix
        log (dict): dictionary where results will be stored
        pred_type (string): string specifying whether X and y are from train, 
                            test or validation set
        config['metrics'] (dict): dictionary specifying which metrics should be calculated and stored.
                        metrics list in config yaml may contain the following:
                        classification:
                            roc_auc
                            precision
                            recall
                            f1score
                            average_precision
                            balanced_accuracy
                            label_distribution
                        regression:
        Returns: 
            log (dict): dictionary with metrics added as key/value pairs.
    '''
    
    # predict on data
    y_pred = trained_model.predict(X)
    
    if config['model_type']=='classification':
        y_pred_prob = trained_model.predict_proba(X)

        # metrics
        # classification

        if config['metrics']['roc_auc']:
            if len(np.unique(y))<=2: 
                log['roc_auc_' + pred_type] = roc_auc_score(y, y_pred_prob[:,1])
            else:
                log['roc_auc_' + pred_type] = np.nan

        if config['metrics']['f1']:
            log['f1_' + pred_type] = f1_score(y, y_pred, average=config['metrics']['multiclass_average_strategy'])

        if config['metrics']['accuracy']:
            log['accuracy_' + pred_type] = accuracy_score(y, y_pred)

        if config['metrics']['precision']:
            log['precision_' + pred_type] = precision_score(y, y_pred, average=config['metrics']['multiclass_average_strategy'])

        if config['metrics']['recall']:
            log['recall_' + pred_type] = recall_score(y, y_pred, average=config['metrics']['multiclass_average_strategy'])

        if config['metrics']['average_precision']:
            if len(np.unique(y))<=2: 
                log['average_precision_' + pred_type] = average_precision_score(y, y_pred_prob[:,1])
            else:
                log['average_precision_' + pred_type] = np.nan

        if config['metrics']['balanced_accuracy']:    
            log['balanced_accuracy_' + pred_type] = balanced_accuracy_score(y,y_pred)

        if config['metrics']['label_distribution']:
            for class_num in np.unique(y):
                log[f'label_count_{str(class_num)}_gt_{pred_type}'] = np.sum(y==class_num)
                log[f'label_count_{str(class_num)}_pred_{pred_type}'] = np.sum(y_pred==class_num)

            
    # regression
    elif config['model_type']=='regression':
        if config['metrics']['explained_variance']:
            log['explained_variance_' + pred_type] = explained_variance_score(y, y_pred)
        if config['metrics']['max_error']:
            log['max_error_ ' + pred_type] = max_error(y, y_pred)
        if config['metrics']['mean_absolute_error']:
            log['mean_absolute_error_' + pred_type] = mean_absolute_error(y, y_pred)
        if config['metrics']['mean_squared_error']:
            log['mean_squared_error_' + pred_type] = mean_squared_error(y, y_pred)
        if config['metrics']['r2']:
            log['r2_' + pred_type] = r2_score(y, y_pred)
    return log
    
