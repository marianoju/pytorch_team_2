from features.preprocessing_imdb import load_data
from models.decision_tree import decision_tree
from evaluation import print_errors_classified

# Daten laden
X_train, X_test, y_train, y_test = load_data()

y_test, y_prediction, model, fit_time, pred_time = decision_tree(
    X_train, X_test, y_train, y_test, max_depth=10, random_state=11)

print_errors_classified(y_test, y_prediction, model, fit_time, pred_time)
