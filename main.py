import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from sklearn.linear_model._cd_fast import ConvergenceWarning
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \
    multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from PreProcessing import PreProcessing

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, vmax, fontsize=14):
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names, )
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes, vmin=0, vmax=vmax,
                              cmap=mpl.colormaps['viridis'])
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("class: " + class_label)


def get_results(model, X, y_true, verbose=False):
    y_prediction = model.predict(X)
    metrics = [balanced_accuracy_score(y_true, y_prediction),
               f1_score(y_true, y_prediction, average='macro')]
    if verbose:
        print('\tbalanced-accuracy: {:0.3f}%'.format(metrics[0] * 100))
        print('\tF1-Score: {:.3f}%'.format(metrics[1] * 100))

        class_id = ['text', 'Horizontal line', 'picture', 'vertical line', 'graphic']
        # print confusion matrix for each class
        mcm = multilabel_confusion_matrix(y_true, y_prediction)
        labels = [class_id[i] for i in range(5)]
        fig, ax = plt.subplots(1, 5, figsize=(10, 2))

        for axes, cfs_matrix, label in zip(ax.flatten(), mcm, labels):
            print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"], len(y_true))

        fig.tight_layout()

        cm = confusion_matrix(y_true, y_prediction, labels=[1, 2, 3, 4, 5])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
    return metrics


def exploratory_data_analysis(df: pd.DataFrame):
    classes = ('text', 'horizzontal line', 'picture', 'vertical line', 'graphic')
    print(df.info())
    ax_bar = plt.subplots()[1]
    ax_heatmap = plt.subplots(figsize=(9, 7))[1]
    classes_counted = [len(df[df["class"] == 1]),
                       len(df[df["class"] == 2]),
                       len(df[df["class"] == 3]),
                       len(df[df["class"] == 4]),
                       len(df[df["class"] == 5])]
    bar_container = ax_bar.bar(classes, classes_counted)
    ax_bar.bar_label(bar_container, fmt='{:,.0f}')
    ax_bar.set(ylabel="instance counted", title="data analysis")

    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, ax=ax_heatmap)
    plt.show()
    return df.drop(['area'], axis=1)


def create_ensemble_and_make_cross_validation(models, X_train, y_train, hparameters, use_grid_search):
    """
    :param models:
    :param X_train:
    :param y_train:
    :param hparameters:
    :param use_grid_search: if true, the function will try searching for the best parameter in hparameters
    :return: ensemble using models list, and the list of the models
    """
    estimator = []
    if use_grid_search:
        for name, model in models.items():
            print('\nGrid search started for {}'.format(name))
            grid_model = grid_search(model, X_train, y_train, hparameters.get(name))
            print('best parameters for model {} are: {}\nmacro-F1-score = {}'.format(name, grid_model.best_params_,
                                                                                     grid_model.best_score_))
            estimator.append((name, grid_model))
    else:
        cross_validate_models(models, X_train, y_train)
        for name, model in models.items():
            estimator.append((name, model))
    # remove Softmax
    estimator.pop(2)
    # remove SVM
    estimator.pop(3)
    clf_stack = StackingClassifier(estimators=estimator, final_estimator=LogisticRegression())
    cross_validate_models({'clf': clf_stack}, X_train, y_train)
    return clf_stack, estimator


def grid_search(model, X_train, y_train, hparameters):
    kfolds = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    y_train = y_train.reset_index(drop=True)
    grid_model = GridSearchCV(estimator=model, param_grid=hparameters, scoring=['f1_macro'],
                              refit='f1_macro', cv=kfolds)
    grid_model.fit(X_train, y_train)
    return grid_model


def cross_validation(model, X_train, y_train):
    """
    :param model: to evaluate
    :param X_train:
    :param y_train:
    :return: accuracy and F1-Score for the passed model
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    y_train = y_train.reset_index(drop=True)
    for train_index, validation_index in skf.split(X_train, y_train):
        model.fit(X_train[train_index], y_train[train_index])
        scores.append(get_results(model, X_train[validation_index], y_train[validation_index], False))
    return scores


def cross_validate_models(models, X_train, y_train):
    """
    Given some models it will train using stratified cross validation
    :param models:
    :param X_train:
    :param y_train:
    :return: Nothing
    """
    print()
    print("Cross validation started")
    for name, model in models.items():
        scores = cross_validation(model, X_train, y_train)
        recall = [item[0] for item in scores]
        f1_scores = [item[1] for item in scores]
        print('{} balanced_accuracy: {:.2f}%, macro-F1_score: {:.2f}%'.format(name, np.mean(recall) * 100,
                                                                              np.mean(f1_scores) * 100))


def create_models(find_hparameters):
    """
    because of the time taken to search for new parameters, if it is not needed just set find_parameters to false
    :param find_hparameters: true if it is needed to search for new parameters
    :return: dict of models and a dict of parameters for grid_search
    """
    models = dict()
    hparameters = {}
    if not find_hparameters:
        # best hyperparameters found yet
        rf = RandomForestClassifier(bootstrap=True, max_depth=10, max_features=7, min_samples_leaf=2,
                                    min_samples_split=3, n_estimators=60)
        softmax = LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='saga', C=1,
                                     penalty=None)
        dt = DecisionTreeClassifier(class_weight='balanced', criterion='entropy')
        knn = KNeighborsClassifier(weights='distance', n_neighbors=5)
        svc = SVC(class_weight='balanced', C=100, gamma=0.001, kernel='linear')
    else:
        rf = RandomForestClassifier()
        softmax = LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='saga')
        dt = DecisionTreeClassifier(class_weight='balanced')
        knn = KNeighborsClassifier(weights='distance')
        svc = SVC(class_weight='balanced')

        param_grid_dt = {'criterion': ['gini', 'entropy']}
        param_grid_Random_forest = {"n_estimators": [50, 60, 100], "max_features": [1, 3, 5, 7],
                                    "max_depth": [3, 5, 10, None], "min_samples_split": [2, 3],
                                    "min_samples_leaf": [1, 2, 3], "bootstrap": [True, False]}
        param_grid_softmax = {'penalty': [None, 'l1', 'l2'], 'C': [1e-5, 5e-5, 1e-4, 5e-4, 1]}
        param_grid_svc = {'C': [1e-4, 1e-2, 1, 1e1, 1e2], 'kernel': ['linear', 'rbf'], 'gamma': [0.001, 0.0001]}
        param_grid_knn = {'n_neighbors': list(range(1, 10, 2))}

        hparameters = {'DecisionTree': param_grid_dt, "RandomForest": param_grid_Random_forest,
                       "Softmax": param_grid_softmax, "SVM": param_grid_svc, 'KNeighbors': param_grid_knn}
    models['DecisionTree'] = dt
    models["RandomForest"] = rf
    models["Softmax"] = softmax
    models["KNeighbors"] = knn
    models["SVM"] = svc
    return models, hparameters


if __name__ == '__main__':

    df = pd.read_csv('data/page_blocks_data.csv')
    df = exploratory_data_analysis(df)

    pre_process = PreProcessing(verbose=True)

    X, y = pre_process.get_feature_target_dataframe(df)
    X_train, X_test, y_train, y_test = pre_process.split_data_in_train_test(X, y, test_ratio=0.20)

    X_train_transf = pre_process.fit_transform(X_train)

    # False because parameters are already calculated
    find_hparameters = False
    models, hparameters = create_models(find_hparameters=find_hparameters)
    clf_stack, estimators = create_ensemble_and_make_cross_validation(models, X_train_transf, y_train, hparameters,
                                                                      use_grid_search=find_hparameters)
    models['clf_stack'] = clf_stack

    if find_hparameters:
        # inserting all models in the dictionary of models, only if it is used grid_search
        # otherwise models dictionary is already complete
        for model in estimators:
            models[model[0]] = model[1]

    print('=' * 60)
    print('\nEvalutation with Test set\n')
    ######################################## Final model ################################################

    print('=' * 60)
    print('\nFinal result are:')

    ## using also the validation set for training
    X_test_transf = pre_process.transform(X_test)

    final_model = models['clf_stack']
    final_model.fit(X_train_transf, y_train)
    get_results(final_model, X_test_transf, y_test, verbose=True)
