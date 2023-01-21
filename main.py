import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data from the "data/hardware" folder
data = pd.read_csv("data/hardware/web_traffic.csv")

# Split the data into training and test sets
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)

# Define the features and target for the model
X_train = train_data[["cpu_spikes", "ram_usage", "http_200", "http_500"]]
y_train = train_data["web_traffic"]

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
X_test = test_data[["cpu_spikes", "ram_usage", "http_200", "http_500"]]
y_test = test_data["web_traffic"]
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Use the model to make a prediction for future web traffic
future_web_traffic = model.predict([[cpu_spikes, ram_usage, http_200, http_500]])

# Suggest number of instances based on the forecasted web traffic
instances = future_web_traffic / 1000 # assume each instance can handle 1000 requests
print("Suggested number of instances:", instances)
