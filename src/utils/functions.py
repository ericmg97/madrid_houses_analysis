from sklearn import metrics
import matplotlib.pyplot as plt

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

def plot_errors(final_model, metric):
    '''
    Plot errors for xgboost models

    Parameters:
        final_model: model to plot errors
        metric: metric to plot

    Returns:
        None
    '''
    
    results = final_model.evals_result()
    epochs = len(results['validation_0'][metric])

    x_axis = range(0,epochs)
    _, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0'][metric], label = 'Train')
    ax.plot(x_axis, results['validation_1'][metric], label = 'Test')
    plt.legend()