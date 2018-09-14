import time


def fit(model, features, targets):
    start = time.time()
    model.fit(features, targets)
    end = time.time()

    print(f'Fit finished in: {end - start}s.')

    return model


def predict(model, data):
    start = time.time()
    prediction = model.predict(data)
    # prediction_probability = model.predict_proba(data)
    end = time.time()

    print(f'Prediction finished in: {end - start}s.')

    return prediction  # , prediction_probability
