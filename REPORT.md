# 4. Model selection and training

## Introduction and problem statement for modeling 

Can we predict whether a given day will be a high-revenue or low-revenue day, based on temporal features?

## Dataset Description after pre-processing 
The dataset contains 9,540 café transactions recorded throughout the full year of 2023 (January 1 to December 31), spread across 20 columns with no missing values. It captures 8 menu items — Juice, Coffee, Salad, Cake, Sandwich, Smoothie, Cookie, and Tea — with Juice being the most frequently purchased. 

Each transaction records the quantity (1–5 units), price per unit (£1–£5), and total spent (£1–£25), with an average transaction value of £8.93 and an average daily revenue of £233.

Orders are split between two location types: Takeaway (70%) and In-store (30%). Payment is made via Digital Wallet (55%), Credit Card (23%), or Cash (23%). Transactions are distributed across all times of day — afternoon (40%), morning (32%), and evening (27%) — and cover both weekdays (71%) and weekends (29%).

Beyond the raw transactional fields, the dataset includes several pre-engineered features such as hour, month, quarter, day_of_week, time_of_day, and binary flags (is_weekend, is_cash, is_credit_card, is_digital_wallet, is_takeaway).
