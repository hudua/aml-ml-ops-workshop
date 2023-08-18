from azureml.core import Workspace, Datastore, Dataset, Model, Run
import pandas as pd
import pickle
import numpy as np
from testmethods import add
import statsmodels

print('version of statsmodels is', statsmodels.__version__)

run = Run.get_context(allow_offline=True)
ws = run.experiment.workspace

x = add.add_example_weird(2, 39)
print("custom module has a weird add method where add(2, 34) equals", x)

dataset = Dataset.get_by_name(ws, name='powergendata')
df = dataset.to_pandas_dataframe()

from sklearn.linear_model import LinearRegression

X = np.array(df[['humidity', 'temperature', 'windspeed']]).reshape(-1, 3)
y = np.array(df['power']).reshape(-1, 1)
model = LinearRegression()
model.fit(X,y)

y_pred = model.predict(X)
abs_error = np.mean(np.abs(y_pred - y))

run.log('abs_error', abs_error)
print('Do not do this in practice, just for the workshop, here is the absolute error:', abs_error)

pickle.dump(model, open('./model.pkl', 'wb'))
model = Model.register(workspace = ws,
                       model_name="mlopsmodeltraining2",
                       model_path = "./model.pkl",
                       description = 'Regression Model'
                      )
