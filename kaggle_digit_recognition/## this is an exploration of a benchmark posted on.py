## this is an exploration of a benchmark posted on Kaggle for the digit recognition challenge, I hope to incorporate this into a meteor site someday #dreamlife
#https://www.kaggle.com/arthurtok/digit-recognizer/interactive-intro-to-dimensionality-reduction

import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

# Import the 3 dimensionality reduction methods
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
