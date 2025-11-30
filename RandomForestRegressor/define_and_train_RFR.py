
# Defining and train RandomForestRegressor
# These hyperparameters are a reasonable starting point.
# You can tune n_estimators, max_depth, etc. later if you have time.

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("Random Forest training complete.")
