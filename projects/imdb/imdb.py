from features.preprocessing_imdb import load_data
from models.dtree.decision_tree import decision_tree
from evaluation import print_errors_classified
from models.ensemble import ensemble_classification

# Daten laden
X_train, X_test, y_train, y_test = load_data()

X_train= X_train[:2000]
X_test = X_test [:2000]
y_test = y_test[:2000]
y_train = y_train [:2000]

y_test, y_prediction, model, fit_time, pred_time = decision_tree(
    X_train, X_test, y_train, y_test, max_depth=10, random_state=11)

print_errors_classified(y_test, y_prediction, model, fit_time, pred_time)

# Ensemble Learning
y_test, en_y_prediction, en_model, en_fit_time, en_pred_time = ensemble_classification(
    X_train, X_test, y_train, y_test, random_state = 11, min_samples_leaf=0.05)

print_errors_classified(y_test, en_y_prediction, en_model, en_fit_time, en_pred_time)
