## Linear and Logistic Regression

### Linear Regression

#### Introduction to Regression

Linear Regression

- is a type of supervised learning model
- models a relationship between continuous target variable and explanatory features
  - X: independent variables
  - y: dependent variables
- can be used to predict a continuous value

What is a regression model?

1. Using features of past cards
2. Build a predictive model
3. That would estimate CO2 emission of a new car

Types of Regression Models

- Simple Regression
  - when an independent variable estimates a dependent variable
  - two types:
    - linear regression
    - nonlinear regression
- Multiple Regression
  - when more than one independent variable estimates a dependent variable
  - two types:
    - linear regression
    - nonlinear regression

Applications of Regression

- want to predict continuous value
- examples:
  - sales forecasting
  - price estimation
  - predictive maintenance
  - employment income
  - rainfall estimation
  - wildfire probability and severity
  - spread of infectious disease
  - risk of chronic disease

Regression Algorithm

- Linear and polynomial
- Random forest
- Extreme gradient boosting (XGBoost)
- K-nearest neighbors (KNN)
- Support vector machines (SVM)
- Neural network

#### Introduction to Simple Linear Regression

Simple Linear Regression

- single independent variable estimates dependent variable
- example:
  - CO2 emission predicted using EngineSize variable

>> add a graph that shows emission in the y axis and engine size in the x axis

- you can determine a best-fit line through the data
- as EngineSize increases, so does CO2 emissions

y_hat = theta0 + theta1 * x1

- y_hat = response variable
- x1 = single predictor
- the model is represented as the equation of a line
- coefficient of linear regression
  - theta0 is the y-intercept
  - theta1 is the slope

Finding the best fit

- given a car with engine size x1 = 5.4
  - the actual co2 emission is 250
  - but, the predicted emission y_hat is 340
  - with a 90 unit discrepancy
- the residual error is the vertical distance between the data points to the fitted regression line
- the average of all residual errors measures how poorly the regression line fits the data, which can be computed by the mean squared error (MSE)
- minimising the mean of the residual errors
- also known as the ordinary least squares

Prediction with linear regression

- you can predict the co2 emission from engine size using the formula

Pros and cons of OLS regression

- OLS regression method is helpful because its easy to understand and interpret
- The method doesn't require any tuning, solutions, just a calculation

#### Introduction to Multiple Linear Regression

Multiple Linear Regression

- extension of simple linear regression model
- uses two or more independent variables to estimate a dependent variable
- measures strength of each independent variable's effect on a dependent variable
- can be used to predict the co2 emissions of an unknown case
- multiple linear regression is a better model than simple linear regression
- however, too many variables can cause overfitting
- to improve prediction, convert categorical independent variables into numerical variables

Regression Model:

y_hat = theta0 + theta1 * x1 + theta2 * x2 + ... + thetan * xn

X = [1, x1, x2, x3, ..., xn] --> feature vectors
theta = --> weights

[theta0]
[theta1]
[theta2]
[...]
[thetan]

y_hat = X * theta

One feature: y = theta0 + theta1 * x1 defines a line
two features y_hat = theta0 + theta1 * x1 + theta2 * x2 defines a plane

independent variables: engine size, cylinders, fuel consumption comb
dependent variable: co2 emissions

Applications of multiple linear regressions

- used in education to predict outcomes and explain relationships
- used to predict the impact of changes in "what-if" scenarios
  - what-if scenario pitfalls
    - inaccurate findings:
      - considering impossible scenarios
      - extrapolating scenarios
      - model might depend on a group of correlated or colinear variables

Colinear pitfalls

- correlated variables are no longer "independent variables"
- if variable correlated with another feature, "what-if" scenario is not possible

Colinear Pitfall Solutions

- remove redundant variables
- select variables which are:
  - most understood
  - controllable
  - most correlated with target

Predictions with multiple linear regression

- multiple linear regression assigns relative importance to features
- choice of features suggests cylinder number has more impact on co2

Fitting a hyperplane

- simple linear regression: regression equation defines a line
- multiple linear regression using two features: solution describes a plane
- beyond two dimensions: solution describes a hyperplane

Model error

- residual error for each car in the data set = difference between its true co2 emission value and predicted value
- average of all residual errors indicates how poorly model predicts the actual values (MSE)

Least squares solution

- factor of 1/n in the MSE equation unnecessary to minimize the error
- method is called least-squares linear regression

Estimating parameters for multiple linear regression

- ordinary least squares estimate coefficients by minimizing MSE
  - this approach uses linear algebra to calculate optimal theta
- gradient descent starts optimization with random values for each coefficient
  - useful for a large data set

#### Polynomial and Non-Linear Regression

Introduction to nonlinear regression

- statistical method for modelling relationship between dependent variable and independent variables
- represented by a nonlinear equation
- equation could be polynomial, exponential, logarithmic, or a nonlinear function
- useful for a complex relationship
- examples:
  - data set with exponential growth pattern
- data has a background trend that follows a smoothed curve
- a smooth, nonlinear curve does a better job at approximating data
- the straight line "under fits" the data

Nonlinear Modeling Techniques

- polynomial regression uses an ordinary linear regression to indirectly fit data to polynomial expressions
- nonlinear regression bases inputs on functions of given features
- such as: logarithm or exponential

Polynomial Regression

- examples:
  - linear
  - quadratic
  - cubic
- relationship between independent variable X and the dependent variable y is modelled as an nth degree polynomial in X

Overfitting with polynomials

- polynomial regression model memorizes everything, including noise or variations
- pick a regression that fits data without overfitting

Applications of nonlinear regression

- nonlinear dependence on input features but linear dependence on regression coefficients
- can be transformed into a linear regression problem
- real-world complex, nonlinear relationships can't be modeled as polynomial
- examples:
  - exponential or compound growth
  - logarithmic
  - periodicity

Compound growth example:

- scatter plot displays strong dependence of GDP on time, but relationship is nonlinear
- GDP increasing over time, but growth rate is also increasing

Productivity by work hours example:

- working more hours per day on average per day increases productivity
- after a reasonable limit, each additional hour generate less productivity

Linear or nonlinear regression

- analyzing scatterplots of target variable against input variable reveal patterns
- express patterns as mathematical functions and determine if:
  - linear
  - exponential
  - logarithmic
  - sinusoidal

Optimizing nonlinear models

- nonlinear machine learning models include:
  - regression trees
  - random forests
  - neural networks
  - support vector machines
  - gradient boosting machines
  - k-nearest neighbors

### Logistic Regression

#### Introduction to Regression

What is logistic regression?

- predicts the probability of an observation belonging to one of two classes
- in machine learning, it refers to a binary classifier based on statistical logistic regression
- observation: true or false (binary classification)
  - predict the probability
    - P(true) >= threshold?
      - TRUE
    - P(true) < threshold?
      - FALSE

When is logistic regression a good choice?

- when the target in data is binary
  - example:
    - yes/no
    - true/false
- when the probability of an outcome is needed
  - logistic regression returns a probability score of sample and maps it to the appropriate class
- if the data is linearly separable, the decision boundary of logistic regression is a line, a plane, or a hyperplane
  - example:
    - theta0 + theta1 * x1 + theta2 * x2 > 0
- to understand the impact of an independent feature
  - select features based on model coefficient size or weights

Logistic Regression Applications

- predicting heart attack risk
- diagnosing patient based on a set of characteristics
- predicting whether a customer will purchase a product or fail a subscription
- predicting product failure probability
- predicting mortgage default likelihood

Model of customer churn data

- Goal: build a model to predict the class of each customer by considering the predicted probability that the customer will churn

Predicting churn using linear regression

- create a rule if y_hat
  - < 0.5 class is 0
  - >= 0.5 class is 1

Towards probabilities

- sigmoid function
  - sigma_x = 1 / (1 + e ^ -x)

Probabilities of class predictions

- p_hat = sigma_y_hat = 1 / (1 + e ^ -y_hat)
  - if sigma_y_hat < 0.5 probability is 0
  - if sigma_y_hat >= 0.5 probability is 1
- 0.5 threshold is called decision boundary

Predicting customer churn

- Churn probability: P(y = 1 | X)
  - P(y = 0 | X) = 1 - P(y = 1 | X)
  - example:
    - P(Churn | Income, age) = 0.8
    - P(Stay | Income, age) = 1 - 0.8 = 0.2

#### Training a Logistic Regression Model

Logistic Regression Training

- identify parameters that map input features to target outcomes
- objective is to predict classes with minimal error
- find parameters or theta that minimizes cost function
- steps:
  1. choose a set of parameters or theta
  2. predict probability that class = 1
  3. calculate prediction error (cost function)
  4. update theta to reduce prediction error
  5. repeat until:
     - reach small log-loss value or
     - targeted number of iterations

Optimal logistic regression

- have a preliminary regression model for the first pass
- uses linear regression to compute y_hat
  - y_hat = theta0 + theta1 * x1
  - p_y_hat = sigma_y_hat = 1 / (1 + e ^ -y_hat)
  - check sigma_y_hat if < 0.5 (class 0) or >= 0.5 (class 1)
- cost function or log-loss needs to be minimized
  - measures how well p_y_hat_i matches y_i

Understanding log-loss

- Confident and correct: predicted probability of class 1 is high and correct => log-loss is small
- Confident and incorrect: predicted probability of class 0 is high and incorrect => log-loss is large

Minimizing cost function with gradient descent

- How?
  - stop training when log-loss is satisfactory
  - use gradient descent
- What is gradient descent?
  - iterative approach to finding the minimum of a function
  - adjusts parameter values using log-loss derivative
  - depends on a specified learning rate
  - controls how far it's allowed to step the parameters

Gradient Descent on a Surface

- Goal: change parameter values and find path to optimal parameters to minimize the cost function
  - best parameters at minimum of cost function
- calculated over the entire data set
- large data set = slow descent
- converge less likely as steps too big to notice minima
- gradient can be approximated using a random subset

Stochastic Gradient Descent (SGD)

- variation of the gradient descent
- uses a random dataset and scales well
- likely to overlook local and find global minima
- converges quickly toward a global minimum
- converge can be improved by:
  - decreasing learning rate
  - gradually increasing sample size
