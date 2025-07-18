import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'Iris flower classification\Data\IRIS.csv')

X = df.drop('species', axis=1)
y = df['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model trained with accuracy: {accuracy:.2f}")

try:
    print("\nEnter the flower measurements:")
    sl = float(input("Sepal Length (cm): "))
    sw = float(input("Sepal Width (cm): "))
    pl = float(input("Petal Length (cm): "))
    pw = float(input("Petal Width (cm): "))

    sample = [[sl, sw, pl, pw]]
    prediction = model.predict(sample)
    species = le.inverse_transform(prediction)

    print("\nPredicted Iris Species:", species[0])

except ValueError:
    print("Invalid input! Please enter numeric values.")
