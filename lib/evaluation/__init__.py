import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, matthews_corrcoef


def print_errors(test, predicted, model, fit_time, pred_time):
    print("")
    print("================================ start ================================")
    print("")
    print(model)
    print("")
    print("Fit finished in: " + str(fit_time) + " s")
    print("Prediction finished in: " + str(pred_time) + " s")
    print("")
    print("MSE: ", mean_squared_error(test, predicted))
    print("RMSE: ", np.sqrt(((test - predicted) ** 2).mean()))
    print("R2: ", r2_score(test, predicted))
    print("RMSE % of mean:", np.sqrt(((test - predicted) ** 2).mean()) / test.mean())
    print("Calibration:", predicted.mean() / test.mean())
    print("")
    print("================================  end  ================================")
    print("")

def print_errors_classified(test, predicted, model, fit_time, pred_time):
    print("")
    print("================================ start ================================")
    print("")
    print(model)
    print("")
    print("Fit finished in: " + str(fit_time) + " s")
    print("Prediction finished in: " + str(pred_time) + " s")
    print("")
    print("R2: ", r2_score(test, predicted))
    print('Matthews (phi) correlation coefficient: ', matthews_corrcoef(test, predicted))
    print("Accuracy Score: ", accuracy_score(test, predicted))
    print("Report: ", classification_report(test, predicted))
    print("")
    print("================================  end  ================================")
    print("")
