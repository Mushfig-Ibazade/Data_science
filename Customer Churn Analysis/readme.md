Churn Dataset Analysis
Introduction
This project aims to analyze a churn dataset and build a predictive model to identify potential churners. Churn refers to customers or users who stop using a service or product. The dataset contains information about various features related to customers, and the target variable indicates whether a customer has churned or not.

Dataset
The dataset used in this project contains the following columns:

Feature 1: Age
Feature 2: Salary
Feature 3: Home_ownership
Feature 4: Employment_time
Feature 5: Loan_purposes
Feature 6: Credit_score
Feature 7: Credit_Amount
Feature 8: Loan_rate
Feature 9: Loan_percentage
Feature 10:Payment_History
Feature 11:Credit_History_Length



Target (Credit_Status): Binary variable indicating churn (1) or non-churn (0).
Please note that the dataset may have a moderate class imbalance, with a higher number of non-churn instances compared to churn instances.

Data Understanding
In the initial phase, the data was explored to gain insights into the dataset. Class distribution and statistical summaries of features were analyzed to understand the data better.

Feature Engineering
To enhance the predictive power of the model, feature engineering techniques were applied. This involved creating new features ,transforming existing ones, and removing irrelevant features that might not contribute significantly to the model's performance.

Data Cleaning
Data cleaning is a crucial step in preparing the dataset for model training. Missing values were handled using a Random Forest imputer, which is an effective method for imputing missing values based on other features in the dataset.

Handling Class Imbalance
As mentioned earlier, the dataset exhibited class imbalance, with the churn class being the minority. To address this issue, various techniques were employed, such as: Weighted Artificial Neural Network and Custimize treshold

Modeling :
Handling the class imbalance in the churn dataset, a weighted neural network model was chosen as it allows assigning higher importance to the minority class during training. 


Conclusion
The churn dataset analysis and predictive modeling allowed us to gain valuable insights into customer behavior and identify potential churners. The deployed model can be used to predict customer churn and implement proactive measures to retain valuable customers.

Deployment part:
The deployment using FastAPI and uvicorn allows for real-time churn predictions via API endpoints. The model can be easily integrated into applications, services, or websites to assist in customer retention efforts.

How to Use the Code
Clone the repository to your local machine.
Ensure you have the required libraries and dependencies installed (list them if any).
Run the provided Jupyter Notebook or Python script for data exploration, feature engineering, and model training.
(Optional) Modify the model hyperparameters or try different algorithms to further improve performance.

Acknowledgments
We would like to acknowledge the source of the churn dataset used in this project and any other relevant data sources.

License
Specify the license under which this code is shared (e.g., MIT, Apache License 2.0).
