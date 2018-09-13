from features.preprocessing_imdb import load_data
from models.dtree.decision_tree import decision_tree_classifier
from models.dtree.random_forest import random_forest_classifier
from evaluation import print_errors_classified
from models.ensemble import ensemble_classification
import pandas as pd
import projects.housing.config as config
import datetime

# Daten laden
X_train, X_test, y_train, y_test = load_data()

# X_train= X_train[:2000]
# X_test = X_test [:2000]
# y_test = y_test[:2000]
# y_train = y_train [:2000]

y_test, y_prediction, model, fit_time, pred_time = decision_tree_classifier(
    X_train, X_test, y_train, y_test, max_depth=10, random_state=11)

print_errors_classified(y_test, y_prediction, model, fit_time, pred_time)


y_test, y_prediction, model, fit_time, pred_time = random_forest_classifier(
    X_train, X_test, y_train, y_test, max_depth=3, random_state=11, n_estimators=40,
    min_samples_leaf=3, warm_start=True)

print_errors_classified(y_test, y_prediction, model, fit_time, pred_time)

# Ensemble Learning
# y_test, en_y_prediction, en_model, en_fit_time, en_pred_time = ensemble_classification(X_train, X_test, y_train, y_test, random_state = 11, min_samples_leaf=0.05)

#print_errors_classified(y_test, en_y_prediction, en_model, en_fit_time, en_pred_time)

df = pd.DataFrame(config.results, columns = ['Accuracy Score', 'Precision Score', 'Recall Score', 'F1-Score', 'Matthews (phi) Score'])
textstring = 'results/results_imdb'+str(datetime.datetime.now())+'.csv'
df.to_csv(textstring, index=False)
print('Results stored in:', textstring)