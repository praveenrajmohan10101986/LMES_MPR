import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd

data = {
    "year": [2000, 2001, 2002, 2003, 2004, 2005],
    "price": [1000, 2000, 3000, 4000, 5000, 6000]
}

# Convert to dataframe
df = pd.DataFrame(data)

x = df[["year"]]
y = df["price"]

Linear_data_model = LinearRegression()
Linear_data_model.fit(x, y)

# Save the model
with open("linear_regression_model_2712.pkl", "wb") as obj:
    pickle.dump(Linear_data_model, obj)

# Below relates to full prediction
data_of_predict = {"year": [2006, 2007]}

df_for_prediction = pd.DataFrame(data_of_predict)

# Below line will take us to previous line which has 2006, 2007 as input and will decide the price
linear_model_prediction = Linear_data_model.predict(df_for_prediction)
print(linear_model_prediction)
