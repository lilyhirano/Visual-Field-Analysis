# Feature importance

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Show top 15 most important features: 

top_k = 15
top_indices = indices[:top_k]

plt.figure(figsize=(8, 6))
plt.bar(range(top_k), importances[top_indices])
plt.xticks(range(top_k), [feature_cols[i] for i in top_indices], rotation=45, ha="right")
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importances for MS Slope Prediction")
plt.tight_layout()
plt.show()

