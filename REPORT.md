# 4. Model selection and training

## Introduction and problem statement for modeling 

The prediction models will focus on answering this question: "Can we predict whether a given day will be a high-revenue or low-revenue day, based on temporal features?" The aim of the prediction would be to understand which time-based factors (such as day of the week, month, seasonality, and holidays) are most strongly associated with revenue fluctuations, and to use these insights to improve planning, forecasting, and decision-making for the cafe business owner.

## Dataset description 

The dataset that has been used for modelin, contains 9,540 café transactions recorded throughout the full year of 2023 (January 1 to December 31), spread across 20 columns with no missing values. It captures 8 menu items (Juice, Coffee, Salad, Cake, Sandwich, Smoothie, Cookie, and Tea) with Juice being the most frequently purchased. 

Each transaction records the quantity (1–5 units), price per unit (£1–£5), and total spent (£1–£25), with an average transaction value of £8.93 and an average daily revenue of £233.

Orders are split between two location types: Takeaway (70%) and In-store (30%). Payment is made via Digital Wallet (55%), Credit Card (23%), or Cash (23%). Transactions are distributed across all times of day, afternoon (40%), morning (32%), and evening (27%) and cover both weekdays (71%) and weekends (29%).

Beyond the raw transactional fields, the dataset includes several pre-engineered features such as hour, month, quarter, day_of_week, time_of_day, and binary flags (is_weekend, is_cash, is_credit_card, is_digital_wallet, is_takeaway).

## First model: Logistic regression 

To address the problem of predicting whether a given day is a high- or low-revenue day, a Logistic Regression model was selected as the baseline classification approach. 

The model estimates the probability that a given day belongs to the high revenue class based on temporal and lag-based features derived from historical data.

We chose to start with this model beacuse: 

- Logistic Regression allows clear interpretation of feature effects, making it suitable for understanding how temporal factors can influence revenue outcomes.
- Works as a strong baseline model against which more complex models can later be compared.
- The model produces probability estimates, enabling analysis of prediction confidence and decision thresholds.

## Training and validation 

A time-based train-test split was applied:

- Training set: January–September 2023
- Test set: October–December 2023

This approach preserves temporal ordering and avoids data leakage, ensuring that the model is evaluated on future, unseen data. However, it would be interesting to see how it performs against other test sets in cross validation folds. Additionally, all input features were standardized using StandardScaler to ensure consistent feature scaling and improve model convergence.

### Results

The Logistic Regression model was evaluated on the test dataset (October–December 2023) to assess its ability to classify days as either high- or low-revenue.

The target variable was well balanced in the test set, with 179 high-revenue days (50%) and 179 low-revenue days (50%). 

This confirms that the use of the median revenue (£230.75) as the threshold successfully avoided class imbalance, ensuring a fair evaluation of model performance. The model achieved an accuracy of 0.609 (60.9%), indicating that it correctly classified approximately 61% of days in the test set.

When the model predicts a low-revenue day, it is correct 62% of the time. However, it only identifies 51% of all actual low-revenue days, meaning it misses nearly half of them. Regarding High revenue days, the model correctly identifies 70% of high-revenue days, with 40% of predicted high-revenue days are incorrect.

The model demonstrates a slight tendency toward predicting high-revenue days more effectively than low-revenue days, therefore the model is better at capturing upward revenue patterns but struggles more with identifying low-demand periods.

The imbalance between recall scores implies that the model may misclassify low-revenue days as high-revenue days, which can lead the busienss to overestimate demand, potentially resulting in overstaffing or excess inventory. To conclude, this model prediction is too optimistic. 

The following code was used: 

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# Load data

df = pd.read_csv('cafe.csv')
df['Transaction Date'] = pd.to_datetime(df['Transaction Date']) # Convert to actual date 

# Aggregation of transaction spent daily 

daily = df.groupby(['Transaction Date', 'day_of_week', 'month', 'quarter', 'is_weekend'])['Total Spent'].sum().reset_index()
daily = daily.sort_values('Transaction Date').reset_index(drop=True)

# Creating lag features to inject temporal memory 

daily['revenue_lag_1']     = daily['Total Spent'].shift(1) # Copy yesterday's revenue into today's row 
daily['revenue_lag_7']     = daily['Total Spent'].shift(7) # Copies revenue from 7 days ago 
daily['revenue_rolling_7'] = daily['Total Spent'].rolling(7).mean() # Calculate average revenue over last 7 days 
daily = daily.dropna().reset_index(drop=True) 

# Create the target variable and binary values 

median_revenue = daily['Total Spent'].median() #Calculate median of all daily revenue 
daily['revenue_class'] = (daily['Total Spent'] > median_revenue).astype(int) # Create column, if below median is low, if above, high 

print(f"Median daily revenue: £{median_revenue:.2f}")
print(f"Class distribution:\n{daily['revenue_class'].value_counts()}")

# Split into train and test periods to train the model 

train = daily[daily['Transaction Date'] < '2023-10-01']
test  = daily[daily['Transaction Date'] >= '2023-10-01']

#Select features to train 

features = ['day_of_week', 'month', 'quarter', 'is_weekend',
            'revenue_lag_1', 'revenue_lag_7', 'revenue_rolling_7']

X_train, y_train = train[features], train['revenue_class']
X_test,  y_test  = test[features],  test['revenue_class']

# Ensure fair coefficient estimation by scaling features

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Model training, find the best weight for each feature 

model = LogisticRegression(random_state=42)
model.fit(X_train_s, y_train)

# Predictions
pred = model.predict(X_test_s)
prob = model.predict_proba(X_test_s)[:, 1]  # probability of high revenue

# Calculate model accuracy 

print(f"\nAccuracy: {accuracy_score(y_test, pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, pred,
      target_names=['Low revenue', 'High revenue']))

# Create confusion matrix
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Low revenue', 'High revenue'])
disp.plot(colorbar=False)
plt.title('Logistic Regression — Confusion Matrix')
plt.tight_layout()
plt.show()

# Plot actual vs predicted over time, success rate
plt.figure(figsize=(14, 4))
plt.plot(test['Transaction Date'].values, y_test.values,
         label='Actual', color='steelblue', marker='o', markersize=3)
plt.plot(test['Transaction Date'].values, pred,
         label='Predicted', color='tomato', linestyle='--', marker='o', markersize=3)
plt.yticks([0, 1], ['Low', 'High'])
plt.title('Logistic Regression — Actual vs Predicted Revenue Class')
plt.xlabel('Date')
plt.ylabel('Revenue Class')
plt.legend()
plt.tight_layout()
plt.show()

# Plot predicted probability over time
plt.figure(figsize=(14, 4))
plt.plot(test['Transaction Date'].values, prob,
         color='purple', marker='o', markersize=3)
plt.axhline(y=0.5, color='red', linestyle='--', label='Decision boundary (0.5)')
plt.title('Logistic Regression — Predicted Probability of High Revenue')
plt.xlabel('Date')
plt.ylabel('Probability')
plt.legend()
plt.tight_layout()
plt.show()

```
```text
Median daily revenue: £230.75
Class distribution:
revenue_class
1    179
0    179
Name: count, dtype: int64

Accuracy: 0.609

Classification Report:
              precision    recall  f1-score   support

 Low revenue       0.62      0.51      0.56        45
High revenue       0.60      0.70      0.65        47

    accuracy                           0.61        92
   macro avg       0.61      0.61      0.60        92
weighted avg       0.61      0.61      0.60        92

```
As shown in the graphs below, there's no meaningful pattern. The model does not reflect "busy periods" vs "quiet periods." Here, the line zigzags unpredictably, which is why accuracy stalled at 64%.


<img width="526" height="470" alt="image" src="https://github.com/user-attachments/assets/ab5c0da6-61b6-4c95-8323-72ceb14561a8" />
<img width="1388" height="390" alt="image" src="https://github.com/user-attachments/assets/a4d824f9-46ea-4300-ba22-e404ef56bd04" />

## Logistic regression with more features 

### Training and validation 

Compared to the previous model, we added more lag features to have more signals on expected revenue and help the model understand time patterns. In this case, incorporating lag features, rolling averages, volatility measures, and seasonal indicators allows the model to learn temporal dependencies, trends, and recurring behaviors in daily revenue. The model was evaluated using a single time-based train/test split rather than cross-validation, this approach was chosen because so the model trains on past data and test on future ones. 

### Results 

The model demonstrates high precision and recall across both classes. It correctly identifies all high-revenue days (recall = 1.00) while maintaining good precision for low-revenue predictions, indicating very few misclassifications overall. However, some features may be very closely related to the target variable in terms of time frame, which can make the model’s performance look better than it really is. This can lead to overly optimistic results that do not reflect how the model will perform in real-world situations. Therefore, even if accuracy appears high, like 98 % in this case, it may not be a reliable indicator of true predictive performance.

The following code was used: 

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# Grouping by total spent daily 
df = pd.read_csv('cafe.csv')
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

daily = df.groupby(['Transaction Date', 'day_of_week', 'month', 'quarter', 'is_weekend'])['Total Spent'].sum().reset_index()
daily = daily.sort_values('Transaction Date').reset_index(drop=True)

# New lag features 
daily['revenue_lag_1']      = daily['Total Spent'].shift(1)
daily['revenue_lag_7']      = daily['Total Spent'].shift(7)
daily['revenue_lag_14']     = daily['Total Spent'].shift(14)   # 2 week lag
daily['revenue_lag_30']     = daily['Total Spent'].shift(30)   # monthly lag

# Detecting trends over time 
daily['revenue_rolling_7']  = daily['Total Spent'].rolling(7).mean()
daily['revenue_rolling_14'] = daily['Total Spent'].rolling(14).mean()  # 2 week trend
daily['revenue_rolling_30'] = daily['Total Spent'].rolling(30).mean()  # monthly trend

# Predicts volatility and direction of change, how unstable is revenue?

daily['revenue_std_7']      = daily['Total Spent'].rolling(7).std()    # how unstable revenue is
daily['revenue_std_14']     = daily['Total Spent'].rolling(14).std()   # 2 week volatility
daily['revenue_momentum']   = daily['Total Spent'].diff(7)             # change vs 7 days ago
daily['revenue_pct_change'] = daily['Total Spent'].pct_change(7)       # % change vs 7 days ago

# Day of the week feature create binary 
daily['is_monday']          = (daily['day_of_week'] == 0).astype(int)
daily['is_friday']          = (daily['day_of_week'] == 4).astype(int)
daily['is_saturday']        = (daily['day_of_week'] == 5).astype(int)
daily['is_sunday']          = (daily['day_of_week'] == 6).astype(int)

# Seasonality features
daily['is_december']        = (daily['month'] == 12).astype(int)  # Christmas season
daily['is_summer']          = (daily['month'].isin([6, 7, 8])).astype(int)

daily = daily.dropna().reset_index(drop=True)

# Use the median to determine revenue class (higer, or lower than median)
median_revenue = daily['Total Spent'].median()
daily['revenue_class'] = (daily['Total Spent'] > median_revenue).astype(int)
print(f"Median daily revenue: £{median_revenue:.2f}")
print(f"Class distribution:\n{daily['revenue_class'].value_counts()}")

# Train and test split 
train = daily[daily['Transaction Date'] < '2023-10-01']
test  = daily[daily['Transaction Date'] >= '2023-10-01']

# Going from 7 to 21 features 

features = [
    # Original features
    'day_of_week', 'month', 'quarter', 'is_weekend',
    # Lag features
    'revenue_lag_1', 'revenue_lag_7', 'revenue_lag_14', 'revenue_lag_30',
    # Rolling averages
    'revenue_rolling_7', 'revenue_rolling_14', 'revenue_rolling_30',
    # Volatility and momentum
    'revenue_std_7', 'revenue_std_14', 'revenue_momentum', 'revenue_pct_change',
    # Day flags
    'is_monday', 'is_friday', 'is_saturday', 'is_sunday',
    # Season flags
    'is_december', 'is_summer'
]

X_train, y_train = train[features], train['revenue_class']
X_test,  y_test  = test[features],  test['revenue_class']

# Coefficient scaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_s, y_train)

# Predictions
pred = model.predict(X_test_s)
prob = model.predict_proba(X_test_s)[:, 1]

# Evaluation 
print(f"\nAccuracy: {accuracy_score(y_test, pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, pred,
      target_names=['Low revenue', 'High revenue']))

# Confusion Matrix 
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Low revenue', 'High revenue'])
disp.plot(colorbar=False)
plt.title('Logistic Regression — Confusion Matrix')
plt.tight_layout()
plt.show()

# Plot actual vs predicted over time
plt.figure(figsize=(14, 4))
plt.plot(test['Transaction Date'].values, y_test.values,
         label='Actual', color='steelblue', marker='o', markersize=3)
plt.plot(test['Transaction Date'].values, pred,
         label='Predicted', color='tomato', linestyle='--', marker='o', markersize=3)
plt.yticks([0, 1], ['Low', 'High'])
plt.title('Logistic Regression — Actual vs Predicted Revenue Class')
plt.xlabel('Date')
plt.ylabel('Revenue Class')
plt.legend()
plt.tight_layout()
plt.show()

# Plot predicted probability over time
plt.figure(figsize=(14, 4))
plt.plot(test['Transaction Date'].values, prob,
         color='purple', marker='o', markersize=3)
plt.axhline(y=0.5, color='red', linestyle='--', label='Decision boundary (0.5)')
plt.title('Logistic Regression — Predicted Probability of High Revenue')
plt.xlabel('Date')
plt.ylabel('Probability')
plt.legend()
plt.tight_layout()
plt.show()
```

```text
Median daily revenue: £230.50
Class distribution:
revenue_class
0    168
1    167
Name: count, dtype: int64

Accuracy: 0.957

Classification Report:
              precision    recall  f1-score   support

 Low revenue       1.00      0.91      0.95        45
High revenue       0.92      1.00      0.96        47

    accuracy                           0.96        92
   macro avg       0.96      0.96      0.96        92
weighted avg       0.96      0.96      0.96        92

```
<img width="526" height="470" alt="image" src="https://github.com/user-attachments/assets/d5a27075-fb83-4771-9d52-8d770e6e706e" />

## Second model: Decision tree  

### Training and validation 

The decision tree model was trained using engineered features including lagged revenue values, rolling averages, and calendar-based variables such as day of the week, month, and weekend indicators. These features were designed to capture both short-term trends and potential seasonal patterns in revenue.

To evaluate the model, a time-based train-test split was used, where the model was trained on earlier observations and validated on later, unseen data. Cross-validation was not used because standard cross-validation techniques randomly shuffle the data, which would break the temporal order and introduce data leakage by allowing future observations to appear in the training set.

### Results 

The decision tree model achieved a training accuracy of 68.8% and a test accuracy of 64.1%, indicating a moderate level of predictive performance with a relatively small gap between training and testing results. This suggests that the model generalizes reasonably well and is not severely overfitting the training data.

Feature importance analysis shows that the model relies primarily on lagged and rolling revenue features (especially revenue_rolling_7 and revenue_lag_1), while calendar-based features such as day of the week, month, and weekend indicators have little to no influence. This indicates that the model is largely driven by short-term revenue trends rather than broader temporal or seasonal effects. To improve performance, further feature engineering could be explored, such as incorporating additional lag periods.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# Prepare data and group into daily totals 
df = pd.read_csv('cafe.csv')
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

daily = df.groupby(['Transaction Date', 'day_of_week', 'month', 'quarter', 'is_weekend'])['Total Spent'].sum().reset_index()
daily = daily.sort_values('Transaction Date').reset_index(drop=True)

# Create lag features
daily['revenue_lag_1']     = daily['Total Spent'].shift(1)
daily['revenue_lag_7']     = daily['Total Spent'].shift(7)
daily['revenue_rolling_7'] = daily['Total Spent'].rolling(7).mean()
daily = daily.dropna().reset_index(drop=True)

# Create target variable finding median revenue
median_revenue = daily['Total Spent'].median()
daily['revenue_class'] = (daily['Total Spent'] > median_revenue).astype(int)

# Split into train and test
train = daily[daily['Transaction Date'] < '2023-10-01']
test  = daily[daily['Transaction Date'] >= '2023-10-01']

features = ['day_of_week', 'month', 'quarter', 'is_weekend',
            'revenue_lag_1', 'revenue_lag_7', 'revenue_rolling_7']

X_train, y_train = train[features], train['revenue_class']
X_test,  y_test  = test[features],  test['revenue_class']

# Train
model = DecisionTreeClassifier(
    max_depth=3,           # keep tree simple, only ask 3 questions 
    min_samples_split=10,  # need 10 days to split
    min_samples_leaf=5,    # each leaf needs 5 samples
    random_state=42
)
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Check overfitting
train_pred = model.predict(X_train)
print(f"Train accuracy: {accuracy_score(y_train, train_pred)*100:.1f}%")
print(f"Test accuracy:  {accuracy_score(y_test, pred)*100:.1f}%")

# Evaluate
print(f"\nAccuracy: {accuracy_score(y_test, pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, pred,
      target_names=['Low revenue', 'High revenue']))

# Confusion matrix
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Low revenue', 'High revenue'])
disp.plot(colorbar=False)
plt.title('Decision Tree — Confusion Matrix')
plt.tight_layout()
plt.show()

#  Actual vs predicted plot
comparison = pd.DataFrame({
    'Date':      test['Transaction Date'].values,
    'Actual':    y_test.values,
    'Predicted': pred,
    'Correct':   (y_test.values == pred)
})

plt.figure(figsize=(14, 4))
for i, row in comparison.iterrows():
    color = 'green' if row['Correct'] else 'red'
    plt.axvline(x=row['Date'], color=color, alpha=0.3, linewidth=2)

plt.plot(test['Transaction Date'].values, y_test.values,
         label='Actual', color='steelblue', marker='o', markersize=4, zorder=5)
plt.plot(test['Transaction Date'].values, pred,
         label='Predicted', color='tomato', linestyle='--', marker='o', markersize=4, zorder=5)
plt.yticks([0, 1], ['Low', 'High'])
plt.title('Decision Tree — Actual vs Predicted (Green = Correct, Red = Wrong)')
plt.xlabel('Date')
plt.ylabel('Revenue Class')
plt.legend()
plt.tight_layout()
plt.show()

# Visualise the tree
plot_tree(model,
          feature_names=features,
          class_names=['Low revenue', 'High revenue'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Structure')
plt.tight_layout()
plt.show()

# Feature importance
importance = pd.DataFrame({
    'Feature':    features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(importance['Feature'], importance['Importance'], color='steelblue')
plt.title('Decision Tree — Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

print(importance)

```
```text
Train accuracy: 68.8%
Test accuracy:  64.1%

Accuracy: 0.641

Classification Report:
              precision    recall  f1-score   support

 Low revenue       0.66      0.56      0.60        45
High revenue       0.63      0.72      0.67        47

    accuracy                           0.64        92
   macro avg       0.64      0.64      0.64        92
weighted avg       0.64      0.64      0.64        92

```

<img width="526" height="470" alt="image" src="https://github.com/user-attachments/assets/9e6c37ff-536a-442c-92bb-04ac5f4fc9be" />

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/f506d54d-55ce-4a96-b17a-5e0d9d7a2e1d" />

# Hyperparameter optimization

## Decision Tree without feature engineering 

### Training and validation 

The decision tree model was trained using the same feature set as the baseline logistic regression model, including basic temporal variables (day of the week, month, quarter, weekend indicator) and simple lag-based features (previous day revenue, 7-day lag, and 7-day rolling average).

These features provide limited short-term memory of past revenue while keeping the model relatively simple and interpretable. Unlike the later experiments, no additional feature engineering such as extended lags, volatility measures, or seasonal flags was applied.

A time-based train-test split was used:

- Training set: January–September 2023  
- Test set: October–December 2023  

This ensures that the model is trained on past observations and evaluated on future data, preserving temporal order and avoiding data leakage. To improve model performance, hyperparameters were tuned using GridSearchCV. The search explored different values for maximum tree depth, minimum number of samples required to split a node, minimum number of samples in leaf nodes, and the splitting criterion (gini or entropy). The best-performing combination of parameters was selected based on cross-validated accuracy.

### Results 

The decision tree model achieved a training accuracy of 68.8% and a test accuracy of 64.1%, indicating moderate predictive performance. The relatively small difference between training and test accuracy suggests that the model generalises reasonably well and is not heavily overfitting.

Although hyperparameter tuning was applied, the improvement over the baseline logistic regression remains limited. This suggests that the main constraint is not the model configuration, but the limited predictive power of the available features. Additionally, since GridSearchCV uses standard cross-validation, it may not fully respect the temporal structure of the data, which can lead to slightly optimistic performance estimates.

Overall, the decision tree captures some short-term patterns but does not significantly outperform simpler models when only basic temporal and lag features are used.

The code, along with the corresponding evaluation metrics and visualisations, is available in `./evaluation/cafemodel2.ipynb` in section `Default features/Decision Tree`.

To further validate the hypothesis that model performance is driven primarily by the choice of features rather than the specific model type or its hyperparameters, we repeat the same analysis using a Random Forest model with hyperparameter optimisation, as described below.


## Random Forest without feature engineering 

### Training and validation 

The Random Forest model was trained using the same feature set as the baseline models, including basic temporal variables (day of the week, month, quarter, weekend indicator) and simple lag-based features (previous day revenue, 7-day lag, and 7-day rolling average).

Random Forest is an ensemble method that builds multiple decision trees on different subsets of the data and combines their predictions. This typically improves robustness and reduces the risk of overfitting compared to a single decision tree.

A time-based train-test split was applied:

- Training set: January–September 2023  
- Test set: October–December 2023  

This ensures that the model is trained on past data and evaluated on future observations, preserving the temporal structure of the dataset.

To optimise performance, hyperparameters were tuned using GridSearchCV. The search explored different values for the number of trees, maximum tree depth, minimum samples required for splits and leaves, and the number of features considered at each split. The best-performing configuration was selected based on cross-validated accuracy.

### Results 

The Random Forest model achieved a training accuracy of 84.6% and a test accuracy of 63.0%. While the training accuracy is notably higher than that of the single decision tree, the test accuracy remains similar, indicating that the ensemble model does not provide a meaningful improvement in generalisation performance.

The classification results show a similar pattern to previous models.

Despite its greater complexity and the use of hyperparameter tuning, the Random Forest model does not significantly outperform simpler models when trained on the same limited feature set. This supports the hypothesis that predictive performance is primarily constrained by the available features rather than the choice of model or its configuration.

Additionally, as with the decision tree, the use of standard cross-validation within GridSearchCV does not fully account for the temporal nature of the data, which may lead to slightly optimistic estimates of model performance.

The code, along with the corresponding evaluation metrics and visualisations, is available in `./evaluation/cafemodel2.ipynb` in section `Default features/Random Forest`.

Overall, the results indicate that increasing model complexity alone is insufficient to achieve better predictions without more informative features.
To further test this hypothesis we perform hyperparmeter optimisations for both Random Forest and Decision Tree with the additional features that were shown to be important previously. The results of these experiments are summarized in the following sections.

## Decision tree with feature engineering  

### Training and validation 

The decision tree model was retrained using an extended set of engineered features to provide richer information about revenue dynamics. In addition to the basic temporal variables (day of the week, month, quarter, weekend indicator), the model includes multiple lag features (1, 7, 14, and 30 days), rolling averages over different time windows, measures of volatility (rolling standard deviations), and indicators of momentum and percentage change.

Additional binary variables were introduced to capture weekly patterns (e.g. specific days of the week) and seasonal effects (e.g. summer months and December). These features allow the model to capture short-term fluctuations, medium-term trends, and recurring seasonal behaviour in the data.

A time-based train-test split was applied:

- Training set: January–September 2023  
- Test set: October–December 2023  

This preserves temporal ordering and ensures that the model is evaluated on future observations.

Hyperparameters were tuned using GridSearchCV. The search explored different values for tree depth, minimum samples required for splits and leaves, and the splitting criterion. The best-performing configuration was selected based on cross-validated accuracy.

### Results 

With the addition of engineered features, the decision tree achieved a training accuracy of 98.8% and a test accuracy of 92.4%, representing a substantial improvement over the version without feature engineering. It is also important to note that both classes are predicted with same quality (identical F1-score, differences in precision/recall less than 3%).

However, the very high training accuracy suggests that the model fits the training data extremely closely. Although test performance is also high, these results should be interpreted with caution. Many of the engineered features (such as recent lags, rolling averages, and momentum) are closely related to the target variable in time, which can make the prediction task easier and potentially lead to overly optimistic performance estimates.

In addition, the use of standard cross-validation within GridSearchCV does not fully respect the temporal structure of the data, which may further contribute to optimistic results.

Overall, the experiment demonstrates that feature engineering has a significantly greater impact on model performance than model complexity. By incorporating richer representations of recent revenue behaviour, the model is able to capture patterns that were not detectable using basic temporal features alone. The resulting accuracy is slightly lower than that obtained with the logostic regression on the same feature set. It confirms the idea that the model performance is almost entirely determined by the feature set and not by the model architecture (algorithm type) and its hyperparameters. 

### Interpretation from an autocorrelation perspective  

A key reason for the strong improvement in model performance after feature engineering lies in the autocorrelation structure of the target variable (daily revenue).

Autocorrelation refers to the relationship between a variable and its past values. In time series data such as daily revenue, it is common for observations to be highly dependent on recent history. For example, revenue on a given day is often influenced by revenue on the previous day, the same day in the previous week, or recent trends over several days.

In this dataset, daily revenue likely exhibits strong short-term autocorrelation. This means that values such as revenue_lag_1 (yesterday’s revenue), revenue_lag_7 (same day last week), and rolling averages contain information that is highly predictive of the current day’s revenue. As a result, when these features are included, the model is effectively given direct signals about the target variable’s recent behaviour.

From this perspective, the classification task becomes much easier. The model is no longer relying only on indirect indicators such as day of the week or month, but instead uses variables that are strongly correlated with the target itself. This explains the large increase in accuracy observed after adding engineered features.

However, this also raises an important concern. When features are highly autocorrelated with the target and very close in time, the model may rely heavily on recent values rather than learning more generalisable patterns. Therefore, the observed behaviour suggests that the target variable is strongly autocorrelated, and that the predictive power of the models is driven primarily by this temporal dependence rather than by broader seasonal or calendar-based effects.

The code, along with the corresponding evaluation metrics and visualisations, is available in `./evaluation/cafemodel2.ipynb` in section `Feature engineering/Decision Tree`.

Finally, we perform the same analysis for Random Forest model.

## Random forest with feature engineering   

### Training and validation 

The Random Forest model was retrained using the extended set of engineered features. In addition to the basic temporal variables (day of the week, month, quarter, weekend indicator), the model incorporates multiple lag features (1, 7, 14, and 30 days), rolling averages over different time windows, volatility measures (rolling standard deviations), and indicators of momentum and percentage change.

Additional binary variables were included to capture weekly patterns (specific days of the week) and seasonal effects (e.g. summer months and December). These features provide the model with richer information about short-term fluctuations, medium-term trends, and recurring seasonal behaviour.

A time-based train-test split was applied:

- Training set: January–September 2023  
- Test set: October–December 2023  

This ensures that the model is trained on past observations and evaluated on future data, preserving the temporal structure of the dataset.

To optimise performance, hyperparameters were tuned using GridSearchCV. The search explored different values for the number of trees, maximum tree depth, minimum samples required for splits and leaves, and the number of features considered at each split. The best-performing configuration was selected based on cross-validated accuracy.

### Results 

With the inclusion of engineered features, the Random Forest model achieved a training accuracy of 100.0% and a test accuracy of 93.5%, making it the best-performing model among all those tested apart from logistic regression described before (the difference is ~2%, which might not be statistically significant and is definitely not important from practical perspective).
The classification results show strong and balanced performance across both classes.

However, the perfect training accuracy suggests that the model fits the training data extremely closely. Although the test performance is also high, these results should be interpreted with caution. Some of the engineered features are closely related to the target variable in time, which may make the prediction task easier and lead to overly optimistic performance estimates.

In addition, the use of standard cross-validation within GridSearchCV does not fully preserve the temporal ordering of the data, which may further contribute to optimistic evaluation results.

Overall, the experiment reinforces that feature engineering has a significantly greater impact on model performance than model choice alone. The Random Forest benefits from the richer feature set, but the performance gains are primarily driven by the additional information provided by the engineered variables rather than the complexity of the model itself.

The code, along with the corresponding evaluation metrics and visualisations, is available in `./evaluation/cafemodel2.ipynb` in section `Feature engineering/Random forest`
# 3. Exploratory Data Analysis
 
The EDA has been performed using a data sample for an entire year containing 9,540 transactions, $85,179.50 revenues, 28,850 units sold, and $8.93 AOV.
 
## 3.1 Temporal Trends
 
Revenue is highly stable throughout the year ($6,600–$7,400 per month), with February showing the lowest figures, and seasonality effects are not prominent.
 
The AOV shows completely independent fluctuations, with maximum and minimum AOV being recorded in April ($9.32) and October ($8.73), respectively, indicating some behavioral changes of customers.
 
Regarding intraday patterns, two peaks in the number of customers have been identified: a morning peak at 07:00 and a higher peak at 13:00 (daily maximum number of 1,291 transactions). Lower traffic and lower AOV are observed during evenings (AOV equals $8.46 at 19:00).
 
Income generation happens evenly throughout the week ($11.7K to $12.4K). Weekdays generate about 71.27% of the income, which is proportionate to the 5:2 ratio, indicating that there is no difference between weekdays and weekends. In the afternoon session, 40.89% of daily income is generated. As per the heatmap, the highest peak time on a daily basis is 13:00. Friday and Saturday lunches yield the most money ($1,780 & $1,910 respectively).
 
## 3.2 Product Mix
 
There are three top categories which include Juice (21.42%), Salad (19.45%) and Sandwich (15.34%), which contribute over 56% towards revenues. Juice has managed to make it because of its volumes (6,100 units @ $2.97). On the other hand, Salad and Sandwich are price and volume driven. Cookie and Tea are found at the bottom in both terms of revenues and margins.
 
## 3.3 Payment Method and Location
 
Digital wallet payment methods are the most popular, accounting for 54.15% of revenue, followed by cash and credit card payments, which account for approximately 23% each.
 
## 3.4 Conclusion
 
The dataset represents a consistent business operation with a high frequency of transactions without any outliers. The demand is highest during lunch time and afternoon hours, as well as in-store purchases. This makes the data suitable for supervised learning algorithms, specifically for predicting revenue based on time and forecasting demand per item.
 
---
 
# Appendix A — EDA Charts
 
### 1. Monthly Revenue, Transactions, and Average Order Value Trend
 
![Monthly Revenue, Transactions, and AOV Trend](assets/media/image1.png)
 
### 2. Revenue and Transactions by Hour of Day
 
![Revenue and Transactions by Hour of Day](assets/media/image2.png)
 
### 3. Top Products by Revenue Contribution and Average Unit Price
 
![Top Products by Revenue Contribution and Average Unit Price](assets/media/image3.png)
 
### 4. Revenue by Payment Method and Location
 
![Revenue by Payment Method and Location](assets/media/image4.png)
 
### 5. Revenue by Time of Day, Day of Week, and Weekday vs Weekend
 
![Revenue by Time of Day, Day of Week, and Weekday vs Weekend](assets/media/image5.png)
 
### 6. Revenue Heatmap: Hour vs Day of Week
 
![Revenue Heatmap: Hour vs Day of Week](assets/media/image6.png)
