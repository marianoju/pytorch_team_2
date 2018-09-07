import time


def fit(model, features, targets):
   start = time.time()
   model.fit(features, targets)
   end = time.time()

   print(f'Training finished in: {end - start}s.')

   return model


def predict(model, data):
   start = time.time()
   pred = model.predict(data)
   pred_proba = model.predict_proba(data)
   end = time.time()

   print(f'Prediction finished in: {end - start}s.')

   return pred, pred_proba
