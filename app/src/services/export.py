def save_predictions(model, dataframe):
    predictions = model.predict(dataframe)
    return predictions, predictions.to_csv(index=False).encode()