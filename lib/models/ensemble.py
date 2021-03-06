from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import time
import evaluation


def ensemble_classification(X_train, X_test, y_train, y_test, *,
                            n_neighbors=5,
                            max_depth=None,
                            min_samples_leaf=0.1,
                            C=1.0,
                            degree=3,
                            gamma='auto',
                            random_state=None):

    # Instantiate LogisticRegression
    lr = LogisticRegression(random_state=random_state)

    # Instantiate KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Instantiate DecisionTreeClassifier
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state)

    # Instantiate SVMC
    svmc = SVC(random_state=random_state, C=C, degree=degree, gamma=gamma)

    # Define the list classifiers
    classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn),
                   ('Classification Tree', dt), ('SVC', svmc)]

    # Instantiate VotingClassifier
    vc = VotingClassifier(estimators=classifiers)
    model = vc

    # Fit vc to the training set
    fit_start = time.time()
    vc.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    # Evaluate the test set predictions
    pred_start = time.time()
    y_pred = vc.predict(X_test)
    pred_end = time.time()
    pred_time = pred_end - pred_start

    evaluation.save_errors_classified(
        y_test, y_pred, model, fit_time, pred_time)
    evaluation.print_errors_classified(
        y_test, y_pred, model, fit_time, pred_time)
