# Makefile for glaucoma-progression-from-vf

# Python environment

ENV=.venv
PYTHON=$(ENV)/bin/python

# Default target

all: install

# Create virtual environment + install requirements

install:
	python3 -m venv $(ENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Environment setup complete."

# Run Jupyter notebook environment

notebook:
	$(PYTHON) -m notebook

# Format code using black

format:
	$(PYTHON) -m black src/ gui/ notebooks/

# Clean Python cache files

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cache cleaned."

# Train severity model
train_severity:

	$(PYTHON) src/models/train_severity.py

# Train progression model
train_progression:

	$(PYTHON) src/models/train_progression.py

# Launch GUI demo (Streamlit)

gui:
	streamlit run gui/app.py

# Remove virtual environment

reset_env:
	rm -rf $(ENV)
	@echo "Environment removed."

