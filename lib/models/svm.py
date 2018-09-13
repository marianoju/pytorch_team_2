from sklearn.svm import SVC,SVR
import time
import evaluation


def svm_regression(X_train, X_test, y_train, y_test, *, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                    tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1):
    svmr = SVR(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C, epsilon=epsilon,
               shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=max_iter)
    svmr_model = svmr
    svmr_fit_start = time.time()
    svmr.fit(X_train, y_train)
    svmr_fit_end = time.time()
    svmr_fit_time = svmr_fit_end - svmr_fit_start

    svmr_pred_start = time.time()
    pred = svmr.predict(X_test)
    svmr_pred_end = time.time()
    svmr_pred_time = svmr_pred_end - svmr_pred_start

    evaluation.save_errors(y_test, pred, svmr_model, svmr_fit_time, svmr_pred_time)

    return y_test, pred, svmr_model, svmr_fit_time, svmr_pred_time


def svm_classification(X_train, X_test, y_train, y_test, *, C=1.0, kernel='rbf', degree=3,
                        gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001,
                        cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                        decision_function_shape='ovr', random_state=None):
    svmc = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,  shrinking=shrinking,
               probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
               verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
               random_state=random_state)
    svmc_model = str(svmc)

    svmc_fit_start = time.time()
    svmc.fit(X_train, y_train)
    svmc_fit_end = time.time()
    svmc_fit_time = svmc_fit_end - svmc_fit_start

    svmc_pred_start = time.time()
    pred = svmc.predict(X_test)
    svmc_pred_end = time.time()
    svmc_pred_time = svmc_pred_end - svmc_pred_start

    evaluation.save_errors_classified(y_test, pred, svmc_model, svmc_fit_time, svmc_pred_time)

    return y_test, pred, svmc_model, svmc_fit_time, svmc_pred_time
