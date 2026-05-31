import itertools
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold


def tune_decision_tree(X_train, y_train, random_state=1):
    """
    train a Decision-Tree using Grid Search and pre-pruning.
    """
    max_depth_list = np.arange(1, 20)
    leaf_samples = np.arange(50, 400, 5)

    # define the parameters to tune
    params_dt = {
        "max_depth": max_depth_list,
        "criterion": ["entropy", "gini"],
        "class_weight": ["balanced", None],
        "min_samples_leaf": leaf_samples,
    }

    tree_model = DecisionTreeClassifier(random_state=random_state)

    stratified_cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=random_state
    )

    grid_search = GridSearchCV(
        estimator=tree_model,
        param_grid=params_dt,
        scoring="roc_auc",
        cv=stratified_cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    result = {
        "model_name": "Decision Tree",
        "trained_model": grid_search.best_estimator_,
        "best_auc": grid_search.best_score_,
        "best_params": grid_search.best_params_
    }

    return result


def tune_mlp(X_train, y_train, random_state=1):
    """
    train MLP using Randomized Search
    """
    model_tune = MLPClassifier(random_state=random_state)

    stratified_cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=random_state
    )

    layer_size = range(1, 6)
    neuron_count = range(2, 100, 5)

    hidden_layer_combinations = []
    for i in layer_size:
        combination = list(itertools.permutations(neuron_count, i))
        hidden_layer_combinations.extend(combination)

    params = {
        "hidden_layer_sizes": hidden_layer_combinations,
        "batch_size": [16, 32, 64, 128, 256, 512],
        "max_iter": [5000],
        "learning_rate_init": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
        "learning_rate": ["constant", "adaptive"],
        "activation": ["logistic", "tanh", "relu"],
        "solver": ["sgd", "adam"],
    }

    random_search = RandomizedSearchCV(
        estimator=model_tune,
        param_distributions=params,
        scoring="roc_auc",
        cv=stratified_cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        random_state=random_state,
        n_iter=100
    )

    random_search.fit(X_train, y_train)

    result = {
        "model_name": "MLP",
        "trained_model": random_search.best_estimator_,
        "best_auc": random_search.best_score_,
        "best_params": random_search.best_params_
    }

    return result


def tune_svm(X_train, y_train):
    """
    linear SVM using RandomizedSearchCV.
    """
    param_dist = {
        "C": [0.1, 1, 5, 10, 50, 80]
    }

    svm_model = SVC(
        kernel="linear",
        probability=True,
        max_iter=2000
    )

    random_search = RandomizedSearchCV(
        estimator=svm_model,
        param_distributions=param_dist,
        scoring="roc_auc",
        cv=10,
        n_iter=10
    )

    random_search.fit(X_train, y_train)

    result = {
        "model_name": "Linear SVM",
        "trained_model": random_search.best_estimator_,
        "best_auc": random_search.best_score_,
        "best_params": random_search.best_params_
    }

    return result


def tune_bagging_decision_tree(X_train, y_train, random_state=1):
    """
    train a Bagging classifer with Decision-Tree base estimator.
    """
    base_estimator = DecisionTreeClassifier(
        min_samples_leaf=50,
        max_depth=14,
        criterion="entropy",
        class_weight="balanced",
        random_state=random_state
    )

    bagging_model = BaggingClassifier(
        estimator=base_estimator,
        random_state=random_state
    )

    param_grid = {
        "n_estimators": range(1, 10)
    }

    stratified_cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=random_state
    )

    grid_search = GridSearchCV(
        estimator=bagging_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=stratified_cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    result = {
        "model_name": "Bagging (ensemble) of Decision Trees",
        "trained_model": grid_search.best_estimator_,
        "best_auc": grid_search.best_score_,
        "best_params": grid_search.best_params_
    }

    return result


def select_best_model(X_train, y_train):
    """
    train model candicates using 10-fold cv. 
    return a dictionary of results, and the best perfomed model based on AUC. 
    """
    all_results = []

    decision_tree_result= tune_decision_tree(
        X_train,
        y_train
    )
    all_results.append(decision_tree_result)


    mlp_result= tune_mlp(
        X_train,
        y_train
    )
    all_results.append(mlp_result)

    svm_result = tune_svm(
        X_train,
        y_train
    )
    all_results.append(svm_result)

    bagging_result= tune_bagging_decision_tree(
        X_train,
        y_train
    )
    all_results.append(bagging_result)

    # find best model based on AUC
    best_result = max(all_results, key=lambda result: result["best_auc"] )

    best_model = best_result["trained_model"]
    best_model_name = best_result["model_name"]

    model_comp_results = []
    for result in all_results:
        model_comp_results.append({
            "model_name": result["model_name"],
            "best_cv_roc_auc": result["best_cv_roc_auc"],
            "best_params": result["best_params"]
        })

    return best_model, best_model_name, model_comp_results

