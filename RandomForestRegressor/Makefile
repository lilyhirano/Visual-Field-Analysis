# Makefile for RandomForestRegressor pipeline (UW glaucoma progression)

PYTHON := python

.PHONY: all rfr train eval importance clean

# Default: run the full RFR pipeline
all: rfr

# Run the main Random Forest script
rfr:
	$(PYTHON) RFR.py

# Optional: run the pieces separately if you want to debug

train:
	$(PYTHON) define_and_train_RFR.py

eval:
	$(PYTHON) evaluate_model.py

importance:
	$(PYTHON) RFR_feature_importance.py

# Clean up temporary files
clean:
	rm -f *.pyc
	rm -rf __pycache__
