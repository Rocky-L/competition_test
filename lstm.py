import os

import pandas as pd
import numpy as np

dataPath = os.path.join(os.getcwd(), 'local.csv')

dateParser = lambda x: pd.datetime.fromtimestamp(float(x))
series = pd.read_csv(dataPath, usecols=['TIME', 'Y'], index_col=0, squeeze=True, parse_dates=['TIME'], date_parser=dateParser)


