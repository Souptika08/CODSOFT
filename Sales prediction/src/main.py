import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"D:\Git\internship\Sales prediction\Data\advertising.csv")
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

tv = float(input("Enter TV advertising budget: "))
radio = float(input("Enter Radio advertising budget: "))
newspaper = float(input("Enter Newspaper advertising budget: "))

features = [[tv, radio, newspaper]]
predicted_sales = model.predict(features)
print(f"Predicted Sales: {predicted_sales[0]:.2f}")
