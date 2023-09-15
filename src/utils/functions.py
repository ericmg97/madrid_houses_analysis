from sklearn import metrics
import pandas as pd

def compute_metrics(model, y_test, y_pred, y_train, y_train_pred):
  '''
    Compute metrics for regression models

    Parameters:
        model: model to compute metrics
        y_test: test target
        y_pred: predicted target
        y_train: train target
        y_train_pred: predicted train target

    Returns:
        None 
  '''

  r2 = metrics.r2_score(y_test, y_pred)
  mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
  mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred)
  mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
  mape_train = metrics.mean_absolute_percentage_error(y_train, y_train_pred)


  print('Train/Test split results:')
  print(model.__class__.__name__+" r2 is %2.3f" % r2)
  print(model.__class__.__name__+" mean_squared_error is %2.3f" % mean_squared_error)
  print(model.__class__.__name__+" mean_absolute_error is %2.3f" % mean_absolute_error)
  print(model.__class__.__name__+" mape test is %2.3f" % mape)
  print(model.__class__.__name__+" mape train is %2.3f" % mape_train)
  
  return r2, mean_squared_error, mean_absolute_error, mape, mape_train

def eval_best_model(final_model, pipeline, valid_df):

    Id_aux = valid_df[['id']]

    X_valid = pipeline.transform(valid_df)
    y_valid_pred = final_model.predict(X_valid)

    submission = pd.DataFrame({'id': Id_aux['id'],
                               'buy_price_by_area': y_valid_pred})
    return (submission)