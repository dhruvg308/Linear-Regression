# Linear Regression From Scratch (NumPy Only)

## Objective
The objective of this assignment is to understand the core concepts of Machine Learning by implementing Linear Regression using Gradient Descent from scratch without using sklearn.

## Approach
In this project, we implemented a simple Linear Regression model using NumPy. We used a small dataset of house area vs price.

Steps followed:
- Data normalization for better convergence
- Initialize weight and bias
- Use Mean Squared Error (MSE) as loss function
- Apply Gradient Descent to update parameters
- Train model over multiple epochs
- Plot Loss vs Epoch graph

## Difficulties Faced
- Gradient Descent was not converging initially
- Loss was increasing due to large values

## Resolutions
- Applied normalization to scale data between 0 and 1
- Adjusted learning rate to a suitable value

## Results
The model successfully learned the relationship between area and price.

Final Learned Equation:
Price = w * Area + b

Loss decreased continuously over epochs which shows proper learning.

## Learnings
- Understood how Gradient Descent works internally
- Learned importance of normalization
- Learned how loss reduces over iterations

## How to Run
1. Install required libraries:
   pip install -r requirements.txt

2. Run the code:
   python linear_regression.py

3. Output:
   - Final equation printed
   - Loss vs Epoch graph displayed

# Linear-Regression
Assignment 1: Linear Regression From Scratch  Objective: Understand core ML fundamentals  
Tasks: 
● Implement Linear Regression using Gradient Descent  
● Train on a dataset (e.g., house prices)  

Constraints:
● Do not use sklearn  
● Use only NumPy  Deliverables:  
● Loss vs Epoch graph  
● Final learned equation
