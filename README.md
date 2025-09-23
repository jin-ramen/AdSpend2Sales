# AdSpend2Sales
# Advertising Spend vs Sales: Predictive Modeling with Linear Regression

## Project Overview

This project demonstrates how to predict **sales revenue** based on advertising spend in different channels (TV, Radio, Newspaper) using *linear regression* trained with *gradient descent*.  
The goal is to build an interpretable model, visualize how advertising budget allocation affects sales, and examine how well simple linear modeling performs on real-world marketing data.

---

## 🌐 Dataset

- **Name:** Advertising Spend vs Sales  
- **Source:** Kaggle — brsahan/advertising-spend-vs-sales :contentReference[oaicite:0]{index=0}  
- **Description:** Contains historical data showing how much money was spent on advertising via TV, Radio, and Newspaper, and what the resulting sales were.  
- **Features:**
  - `TV` — advertising budget for television  
  - `Radio` — advertising budget for radio  
  - `Newspaper` — advertising budget for newspapers  
  - `Sales` — target variable (sales revenue)  
- **Size:** (mention number of rows here after loading, e.g. 200 observations)  

---

## 🎯 Problem Statement

> Can the amount of money spent in different advertising channels explain and predict sales? Specifically, how well can a linear model (using gradient descent) approximate the relationship between ad spend and sales?

Key questions explored:
- Which advertising channel has the strongest effect on sales?
- How well does a linear model perform in predicting sales?
- How many iterations / what learning rate are needed for stable convergence?
- What are the limitations of using linear regression in this domain?

---

## 🧮 Approach / Methodology

1. **Data Exploration & Cleaning**
   - Load CSV, inspect missing values, distribution of feature and target variables.  
   - Visualize pairwise relationships (scatter plots) to check linearity assumptions.

2. **Feature Preparation**
   - (Optional) Scaling or normalization (e.g. standardization) if needed for gradient descent stability.  
   - Add an intercept (bias) feature to the feature matrix.

3. **Model Implementation**
   - Implement linear regression using *batch gradient descent*.  
   - Cost function: Mean Squared Error (MSE).  
   - Track cost over iterations to monitor convergence.

4. **Model Training & Hyperparameter Tuning**
   - Try different learning rates (α) and number of iterations.  
   - Plot cost vs iteration to choose settings that converge well without overshooting.

5. **Evaluation**
   - Compare predictions vs actual sales (scatter plot, line of best fit).  
   - Compute error metrics: MSE, MAE.  
   - Inspect residuals to check for bias or heteroscedasticity.

6. **Interpretation & Insights**
   - Examine learned model coefficients to see which ad channel(s) have strongest influence.  
   - Discuss if the effects seem reasonable (e.g. more TV spend = more sales?).  
   - Highlight limitations and possible improvements (nonlinear models, interaction terms, more features).

---

## 📐 Implementation Details

- **Languages / Tools:** Python, Pandas, NumPy, Matplotlib / Seaborn (for plotting)  
- **Structure:**
