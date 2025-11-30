import pandas as pd
from unsupervised_model import Unsupervised_Model
from analyze_uns_model import Analyzer

data = pd.read_pickle('./pkl_data/GRAPE.pkl')
grape_model = Unsupervised_Model(data)
grape_model.train(n=7, c=4)
grape_model.make_visual()
grape_model.metrics()

analyzer = Analyzer(grape_model.labels, grape_model.patients, grape_model.masks)

analyzer.run_all()