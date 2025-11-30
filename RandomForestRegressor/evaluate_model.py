# Evaluating the model

y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f} dB/year")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} dB/year")
print(f"R^2 Score: {r2:.4f}")

