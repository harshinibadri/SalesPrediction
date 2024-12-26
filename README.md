### **Project Documentation: Sales Prediction Model for Car Purchase Amount**

---

### **1. Overview of the Problem**

The task is to predict the **car purchase amount** for customers based on their demographic and financial attributes. The objective is to build a model that can forecast how much a customer will spend on a car, depending on factors such as their **age**, **gender**, **annual salary**, **credit card debt**, **net worth**, and **country**. 

This type of prediction can be highly useful for businesses to optimize marketing strategies, target the right audience, and improve customer relationships.

---

### **2. Dataset Description**

The dataset contains information about customers, including both their demographic and financial details. Below are the columns in the dataset:

- **customer name**: The name of the customer (irrelevant for prediction and dropped).
- **customer e-mail**: The email of the customer (irrelevant for prediction and dropped).
- **gender**: The gender of the customer (numeric, where 1 represents male and 0 represents female).
- **age**: The age of the customer (numerical).
- **annual Salary**: The annual salary of the customer (numerical).
- **credit card debt**: The total credit card debt of the customer (numerical).
- **net worth**: The net worth of the customer (numerical).
- **country**: The country where the customer resides (categorical).
- **car purchase amount**: The target variable representing the amount spent by the customer on purchasing a car (numerical, to be predicted).

---

### **3. Data Preprocessing and Transformation**

#### **3.1 Handling Irrelevant Features**

- The columns **'customer name'** and **'customer e-mail'** are dropped because they are not related to predicting the car purchase amount. These columns do not contain useful information for the model.

#### **3.2 Splitting Data into Features and Target**

- The dataset is split into two parts:
  - **Features (X)**: These include all the columns except **'car purchase amount'**. The features used for prediction are:
    - **gender**: Represents the gender of the customer (binary: 0 or 1).
    - **age**: Numerical value representing the age of the customer.
    - **annual Salary**: Numerical value representing the annual salary.
    - **credit card debt**: Numerical value representing the total credit card debt.
    - **net worth**: Numerical value representing the net worth of the customer.
    - **country**: Categorical value representing the country the customer resides in.
  - **Target (y)**: This is the **'car purchase amount'**, which is the amount a customer spends on purchasing a car.

#### **3.3 Preprocessing Numerical and Categorical Data**

- **Numerical Data**: 
  - The numerical columns are: **'age'**, **'annual salary'**, **'credit card debt'**, and **'net worth'**.
  - These features are normalized using **StandardScaler**. This ensures that all numerical features are on the same scale, which is important for many machine learning algorithms. Standardization transforms the data such that it has a mean of 0 and a standard deviation of 1.

- **Categorical Data**:
  - The **'country'** column is categorical (with possible values like 'USA', 'Brazil', etc.), so it needs to be transformed into a numerical format that the model can understand.
  - **OneHotEncoder** is used to convert the categorical column into binary columns. For example, if there are 3 unique countries (USA, Brazil, and UK), this will create 3 new columns where each column corresponds to whether a customer belongs to that country (1 or 0).
  
  After encoding, the country data is represented as multiple binary columns, one for each possible country value.

#### **3.4 Combining Preprocessing with the Model**

- To streamline the process, a **Pipeline** is used to ensure that the preprocessing steps and the model training occur in a consistent order.
  - The **ColumnTransformer** handles both the normalization of numerical features and the one-hot encoding of categorical features.
  - After preprocessing, the **RandomForestRegressor** is applied to the data to build the predictive model.

---

### **4. Splitting Data into Training and Test Sets**

- The data is split into training and testing sets using an **80/20 split**:
  - **Training Set (80%)**: This portion is used to train the model and learn the relationships between the features and the target variable.
  - **Test Set (20%)**: This portion is kept aside to evaluate the performance of the trained model and to check if it generalizes well to unseen data.

---

### **5. Hyperparameter Tuning with RandomizedSearchCV**

To optimize the performance of the **RandomForestRegressor**, we perform hyperparameter tuning using **RandomizedSearchCV**. This process tests various combinations of hyperparameters to identify the best settings for the model. The key hyperparameters tuned include:

- **n_estimators**: Number of trees in the forest. More trees often lead to better model performance but at the cost of increased computation.
- **max_depth**: Maximum depth of each tree. A deeper tree can model more complex relationships but may also lead to overfitting.
- **min_samples_split**: The minimum number of samples required to split an internal node. This prevents the model from learning overly specific patterns that might not generalize well.
- **min_samples_leaf**: The minimum number of samples required to be at a leaf node.
- **bootstrap**: Whether to use bootstrap sampling when building trees. This controls whether each tree gets trained on a random subset of the data.

By searching over these parameters, the best combination is found, which helps in improving the model’s accuracy.

---

### **6. Model Evaluation**

After training the model with the optimal hyperparameters, the model's performance is evaluated on the test set using the following metrics:

- **R² Score**: This is the coefficient of determination, which indicates how well the model's predictions match the true values. An R² score closer to 1 means the model has high predictive power.
- **Mean Absolute Percentage Error (MAPE)**: This metric measures how far the predictions are from the true values, expressed as a percentage. A lower MAPE indicates better accuracy.

---

### **7. Example Prediction**

To demonstrate the model’s capability, an example prediction is made for a customer with specific attributes:
- **Country**: Brazil
- **Gender**: Male (represented as 1)
- **Age**: 40 years
- **Annual Salary**: $60,000
- **Credit Card Debt**: $5,000
- **Net Worth**: $300,000

The model predicts the **car purchase amount** based on these attributes.

---

### **8. Conclusion**

- The RandomForestRegressor, after hyperparameter tuning and preprocessing, successfully predicts the car purchase amount with a good **R² score** and a low **MAPE**, indicating the model’s high accuracy.
- The **data preprocessing steps** like **scaling numerical features** and **encoding categorical variables** ensure that the model receives clean, well-formatted data, which improves its performance.
- The model can be used for real-world predictions, where businesses can input customer data to predict how much a customer is likely to spend on a car, allowing for better targeting of marketing efforts.


