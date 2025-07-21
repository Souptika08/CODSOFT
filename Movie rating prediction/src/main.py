import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    print("Loading dataset...")
    path = r'D:\Git\internship\CODSOFT\Movie rating prediction\Data\IMDb Movies India.csv'
    df = pd.read_csv(path, encoding='ISO-8859-1')

    print("Dataset loaded. Shape:", df.shape)

    df = df.dropna(subset=['Rating'])

    df['Duration'] = df['Duration'].str.extract(r'(\d+)')
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

    required_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Rating']
    df = df[required_cols].dropna()

    if df.empty:
        print("No rows left after cleaning.")
        return

    print("Cleaned data. Rows:", df.shape[0])

    print("\nShowing data visualizations...")

    # Rating distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Rating'], bins=20, kde=True, color='skyblue')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Duration vs Rating scatter plot  
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x='Duration', y='Rating', data=df)
    plt.title('Duration vs Rating')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Rating')
    plt.tight_layout()
    plt.show()

    df_encoded = pd.get_dummies(df, columns=['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], drop_first=True)

    X = df_encoded.drop(columns=['Rating'])
    y = df_encoded['Rating']

    if X.empty:
        print("Feature matrix is empty. Cannot train model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nModel trained successfully!")
    print(f"Mean Squared Error (MSE): {mse:.4f}")

if __name__ == "__main__":
    main()
