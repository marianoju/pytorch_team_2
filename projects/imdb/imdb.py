from features.preprocessing_imdb import load_data
from models.dtree.decision_tree import decision_tree_classifier
from models.dtree.random_forest import random_forest_classifier
from models.ensemble import ensemble_classification
import pandas as pd
import projects.globalvar as globalvar
import datetime

# Daten laden
X_train, X_test, y_train, y_test = load_data()

X_train = X_train[:2000]
X_test = X_test[:2000]
y_test = y_test[:2000]
y_train = y_train[:2000]

decision_tree_classifier(X_train, X_test, y_train, y_test,
                         max_depth=10,
                         random_state=11)

random_forest_classifier(X_train, X_test, y_train, y_test,
                         max_depth=3,
                         random_state=11,
                         n_estimators=40,
                         min_samples_leaf=3,
                         warm_start=True)

# Ensemble Learning
ensemble_classification(X_train, X_test, y_train, y_test,
                        random_state=11, min_samples_leaf=0.05)  # noqa: E501


df = pd.DataFrame(globalvar.results, columns=[
                  'Accuracy Score',
                  'Precision Score',
                  'Recall Score',
                  'F1-Score',
                  'Matthews (phi) Score'])
filename = 'results/results_imdb_' \
           + datetime.now().strftime("%Y-%m-%d_%H%M%S_%u") \
           + '.csv'
df.to_csv(filename, index=False)
print('Results stored in:', filename)
