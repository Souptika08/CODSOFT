# Sales Prediction using Python

This project predicts product sales based on advertising budgets using machine learning techniques. It utilizes the **Advertising dataset**, which includes marketing expenditures across **TV**, **Radio**, and **Newspaper** channels. The goal is to help businesses forecast sales and make smarter advertising investment decisions.

---

## Features Used

The model uses the following features (independent variables) to predict sales:

- `TV` – Budget spent on TV advertisements
- `Radio` – Budget spent on radio advertisements
- `Newspaper` – Budget spent on newspaper advertisements

The target variable is:

- `Sales` – Units sold (in thousands)

---

## Algorithm Used

- **Linear Regression** (from `scikit-learn`):
  - A supervised learning algorithm that models the relationship between input features and the output variable by fitting a linear equation to the observed data.

---

## Project Structure

### Folder/Files Explained:

- `Data/advertising.csv` → Dataset used for training and testing  
- `src/main.py` → The core script for:
  - Accepting user input for TV, Radio, and Newspaper ad budgets
  - Uses the trained model to predict expected **Sales**
  - Displaying the result in real time 
- `Notebook/` → Jupyter Notebook for experimenting with datas and model like:
  - Loads and visualizes the dataset
  - Performs exploratory data analysis (EDA)
  - Trains a Linear Regression model
  - Evaluates the model using metrics like **R² score** and **Mean Squared Error**  
- `requirements.txt` → List of required Python packages  
- `README.md` → Project description and instructions

---