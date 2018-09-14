import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, \
    classification_report, matthews_corrcoef, \
    precision_score, recall_score, f1_score
import projects.housing.config as config


def print_errors(test, predicted, model, fit_time, pred_time):
    print('')
    print('========================== start ==========================')
    print('')
    print(model)
    print('')
    print('Fit finished in: ' + str(fit_time) + 's')
    print('Prediction finished in: ' + str(pred_time) + 's')
    print('')
    print('MSE: ', mean_squared_error(test, predicted))
    print('RMSE: ', np.sqrt(((test - predicted) ** 2).mean()))
    print('R2: ', r2_score(test, predicted))
    print('RMSE % of mean:', np.sqrt(
        ((test - predicted) ** 2).mean()) / test.mean())
    print('Calibration:', predicted.mean() / test.mean())
    print('')
    print('==========================  end  ==========================')
    print('')


def print_errors_classified(test, predicted, model, fit_time, pred_time):
    print('')
    print('========================== start ==========================')
    print('')
    print(model)
    print('')
    print('Fit finished in: ' + str(fit_time) + 's')
    print('Prediction finished in: ' + str(pred_time) + 's')
    print('')
    print('R2: ', r2_score(test, predicted))
    print('Matthews (phi) correlation coefficient: ',
          matthews_corrcoef(test, predicted))
    print('Accuracy Score: ', accuracy_score(test, predicted))
    print('Report: ', classification_report(test, predicted))
    print('')
    print('==========================  end  ==========================')
    print('')


def save_errors(test, predicted, model, fit_time, pred_time):

    mse = mean_squared_error(test, predicted)
    rmse = np.sqrt(((test - predicted) ** 2).mean())
    r2 = r2_score(test, predicted)
    rmseom = np.sqrt(((test - predicted) ** 2).mean()) / test.mean()
    cal = predicted.mean() / test.mean()

    result = [model, fit_time, pred_time, mse, rmse, r2, rmseom, cal]
    config.results.append(result)


def save_errors_classified(test, predicted, model, fit_time, pred_time):

    matthewscorr = matthews_corrcoef(test, predicted)
    accuracyscore = accuracy_score(test, predicted)
    precisionscore = precision_score(test, predicted)
    recallscore = recall_score(test, predicted)
    f1score = f1_score(test, predicted)

    result = [accuracyscore, precisionscore,
              recallscore, f1score, matthewscorr]
    config.results.append(result)