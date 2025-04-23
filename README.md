# DS-Encyclopedia
Descriptive and Representational Model:
Descriptive: Based on observed data to capture patterns, explain observed features and relationships(summarization, stats summaries mean, median, linear regression, logistic regression)
Representational: Create underlying mechanism using flow charts, entity relationship diag, like simulate system behavior
 
Analysis:
--------f
Type of Analysis	Question it Answers		Example
Descriptive		What and when?			Sales figures by product category, time period
Example: A retail store might use descriptive analytics to track sales figures by product category or time period. They could see that sales of winter coats spiked in November or that a particular brand of sneakers is consistently popular.

Diagnostic		Why?				Reasons for sales trends
Example: Continuing with the retail store, they might use diagnostic analytics to understand why winter coat sales spiked in November. Was it due to a marketing campaign, a cold snap, or a competitor's price increase? 

Predictive		What might happen next?		Future demand for specific products
Example: The store might use predictive analytics to forecast future demand for specific products based on weather patterns, upcoming holidays, and customer purchasing trends. This allows them to optimize inventory levels and avoid overstocking or understocking.

Prescriptive		What should I do?		Recommended actions to optimize outcomes
Example: Based on the predictive analysis, the store might prescribe targeted marketing campaigns for winter coats in warmer regions or suggest offering discounts on less-popular brands to free up inventory space.

Distribution:
------------
Probability distribution: Discrete like 1,2,3 etc... Continuous like values between [0,5] 
Distribution Types: 
Emperical Distributions:  Based on actual data collected. Collected from existing data from systems or use computers to generate the data/values
Example: Imagine measuring the heights of 100 students. The empirical distribution would show the specific frequency of each height observed (e.g., 5 students at 160 cm, 10 students at 170 cm, etc.).
Standard Distributions: Based on a theoretical mathematical model, such as the normal (Gaussian) distribution, binomial distribution, or Poisson distribution.
Example: The normal distribution, often represented by a bell curve, is a common standard distribution used to model various continuous variables like human heights, test scores
Binomial:Models the probability of success (or "heads") in a fixed number of independent trials, each with only two possible outcomes (success or failure).
Uniform:Assumes all outcomes within a specific range are equally likely. Rolling a fair die. Each face has a 1/6 probability of landing on top.


***Predictive Model leverage the Probability distribution to identify the parameters with range of values (Beta [uniform dist] and Gamma [triangular dist]) with input variables to drive the output

Using Probability distribution for informed decisions

Binomial Distribution Explained with PMF, CDF, and PPF in Python:
The binomial distribution models the probability of success in a series of independent trials with two potential outcomes
1. Probability Mass Function (PMF):
The PMF gives the exact probability of getting a specific number of successes (heads) in the trials.
2. Cumulative Distribution Function (CDF):
The CDF gives the probability of getting at most a certain number of successes (heads) in the trials.
3. Percent Point Function (PPF): (Inverse of CDF)
The PPF (also known as quantile function) tells you the number of successes (heads) required to achieve a desired cumulative probability.

from scipy.stats import binom
n = 10  # Number of trials
p = 0.5  # Probability of success (heads)
# Calculate PMF for different numbers of heads (k)
k_range = range(n+1)  # Possible outcomes (0 to 10 heads)
pmf_values = binom.pmf(k_range, n, p)
cdf_values = binom.cdf(k_range, n, p)
desired_prob = 0.75
ppf_value = binom.ppf(desired_prob, n, p)


***Probabilistic modeling within linear regression to handle uncertainty and estimate prediction intervals.
import numpy as np
from matplotlib import pyplot
from numpy.random import normal
from numpy import hstack
# generate a sample
sample1 = normal(loc=80, scale=5, size=300)  #generate sample using normal dist with mean 80 and SD 5 with size of 300
sample2 = normal(loc=60, scale=5, size=700)
sample = hstack((sample1, sample2))  #combined horizontally

from statsmodels.distributions.empirical_distribution import ECDF
ecdf = ECDF(sample)
data = [["P(x<94): ", ecdf(94)], ["P(x>90): ", 1 - ecdf(90)]]  #creating probability of random variable with <94% and > 90% of empirical dist

Coefficient of Variance= Mean/Stand Dev.

Portfolio Stock Investment:
--------------------------
Mean: Return of investment :z=aX+bY  (where a = fraction investment made, b=1-a; X= mean of returns, Y=std Dev(risk)) 
Variance: sqrt of (aP)^2+(bQ)^2  (where P and Q are SD's of investments) if corr=0 
Variance: sqrt of (aP)^2+(bQ)^2 +2ab*PQ*CORR(X,Y) (where P and Q are SD's of investments) if corr >0 [Lower the correlation i.e, negative correlation, lower the SD and risk, therefore lower Correlation(and zero variance)]

**a positive correlation generally leads to a lower variance between the cobehavior of random variables as it applies to only linear relationship

Variance: Ssquare =sum of squares of Xi-Xmean /n
Co-Variance: Sx,y=Sum of (Xi-Xmean)*(Yi-Ymean)/n
Correlation : Rx,y = Sx,y/SDx*SDy  (unitless value between -1 and 1)
* concept of Covariance and Correlation factors capture how closely data points are correlated linearly

A = [45,37,42,35,39]
B = [38,31,26,28,33]
C = [10,15,17,21,12]
covMatrix = np.cov(X, bias=False)

data1 = 20 * rnd.randn(1000) + 100 #generates 1000 random numbers from a standard normal dist, bell-shaped with a mean of 0 and a standard deviation of 1.
data2 = (data1 * .2 + (10 * rnd.randn(1000) + 500))

from numpy import loadtxt
# load the data using numpy's loadtext. Assign the output to variable "data". 
# The data has 2 columns, we will create a scatterplot with the data. 
data=loadtxt("data.csv",  delimiter=',') #find if there is correlation between the amount in advertising spent vs the total sales for the day.
fig = pyplot.scatter(data[0], data[1])
points = fig.get_offsets()
pyplot.show()


**Observation: describe certain objects(customers, projects etcc.)
Data could be descriptive features like (attributes), outcome features (driven by performance measures) leads to classification labels(binary prediction)

Clustering: Aim to divide group of observations into similar groups/objects
----------
unsupervised learning had no access to outcome featurees or lables. only access to descriptive features
Personalization: Benefits of clustering:
Similar objects behave similar manner	: apply same marketing approach to all sub groups
Similar objects work well together	: deciding on how to form teams
Similar Ojects have same level of risk	: deciding what kind of insurance policies to offer

Objects with attrs + Hyperparameters -> Clusters/Cluster metrics (similarity/dissimilarity measures within and without clusters)

In a plane, you calculate the centroid of multiple objcets/observations. You can calculate by (mean of Xpoints, mean of Ypoints)
**Euclidean distance is used to measure distance between points in 2D like pythagores theorem sqrt of (X1-Y1)square + (X2-Y2)square
**Mahattan distance is absolute value sum of Xi-Yi (all points)
**Maximum distance is maximum of all coordinates Xi-Yi

Normalization:
-------------
df = pd.DataFrame([[180000, 110, 18.9, 1400], [360000, 905, 23.4, 1800]], columns=['Col A', 'Col B'])
df_max_scaled = df.copy()
# apply normalization techniques using MAX SCALING technique
for column in df_max_scaled.columns:
    df_max_scaled[column] = df_max_scaled[column]  / df_max_scaled[column].abs().max()

df_min_max_scaled = df.copy()
# apply normalization techniques using MIN-MAX feature technique
for column in df_min_max_scaled.columns:
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    
 
df_z_scaled = df.copy()
# apply normalization techniques using STANDARDIZATION
for column in df_z_scaled.columns:
    df_z_scaled[column] = (df_z_scaled[column] - df_z_scaled[column].mean()) / df_z_scaled[column].std()    

K-Means:
-------
Showcasing Simple K-Means Clustering Project for Customer Segmentation
https://www.youtube.com/watch?v=HpMhRXrOLWY
https://www.analyticsvidhya.com/blog/2021/05/k-means-clustering-with-mall-customer-segmentation-data-full-detailed-code-and-explanation/

Input observation(data point), hyperparameter set number of clusters 'k', algorithm creates the centroid. Assign each observation to nearest centroid. Recalcuate the k centroids as the avg of their assigned observations. Repeat until convergence
It is a centroid based algorithm in which each cluster is associated with a centroid
The algorithm takes raw unlabelled data as an input and divides the dataset into clusters and the process is repeated until the best clusters are found.

Agglomerative hierarchical clustering (AHC):
-------------------------------------------
AHC stands for Agglomerative Hierarchical Clustering, which is a bottom-up approach to clustering.
The height of the nodes in the dendrogram represents the dissimilarity or distance between the merged clusters.

from scipy.cluster.hierarchy import dendrogram, linkage
# Perform hierarchical clustering
linked = linkage(data, method='ward')
# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked, orientation='top', labels=range(1, 11), distance_sort='descending', show_leaf_counts=True)

The Gaussian distribution: also known as the normal distribution, is a continuous probability distribution that is symmetric around its mean
-------------------------
Bayes' theorem: is a fundamental concept in probability theory that describes the probability of an event based on prior knowledge of event
--------------
data containing 2 components with random normal distribution of numpy
# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(data)

# Cluster assignment for a new data point using Bayes theorem
new_data_point = np.array([[7, 8]])
cluster_probabilities = gmm.predict_proba(new_data_point)
 
##########MATPLOT PLOT############
import matplotlib.pyplot as plt
# Create a scatter plot using Matplotlib
plt.figure(figsize=(8, 6))
plt.scatter(data_cluster1[:, 0], data_cluster1[:, 1], label='Cluster 1', alpha=0.7, color='blue')

# Add labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Synthetic Data - Cluster 1')

# Show the plot
plt.legend()
plt.show()

df.plot(kind = 'bar')
##########MATPLOT PLOT############
##########MATPLOT SEABORN PLOT############

import seaborn as sns
# Create a scatter plot using Seaborn
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_cluster1[:, 0], y=data_cluster1[:, 1], label='Cluster 1', alpha=0.7, color='blue')

# Add labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Synthetic Data - Cluster 1')

# Show the plot
plt.legend()
plt.show()
##########MATPLOT SEABORN PLOT############

PCA:
----
Principal Component Analysis (PCA) is a dimensionality reduction technique commonly used in data analysis and machine learning. It aims to transform the original variables into a new set of uncorrelated variables, called principal components, that capture the maximum variance in the data
Apply covariance on data, then perform eigendecomposition on the covariance matrix to find the eigenvalues and eigenvectors. The eigenvectors represent the directions of maximum variance, and the eigenvalues indicate the magnitude of the variance in those directions.
Example:
Let's say your 10 features are: height, weight, hair color, eye color, glasses, beard, favorite music, favorite food, favorite movie, and shoe size. PCA might find that the main components are:
Component 1: "Tall and sporty" (combines height, weight, and favorite music).
Component 2: "Creative and stylish" (combines glasses, beard, favorite movie, and shoe size).
By focusing on these components, you can quickly understand who shares similar interests and appearances without needing to memorize every individual detail.
Remember, PCA is like a memory trick for data. It helps you identify the key information and discard the less relevant, making complex data easier to handle and understand.


LINEAR REGRESSION:
------------------
skim(df) --> discovery function provides column uniqueness, #of rows/columns, missing, types of variables etc..
*est = ols(formula="rentals ~ temp", data=df).fit()  -->linear regression number of regressions of the type 'rentals' vs. independent variable (eg. temp)
est  = ols(formula="rentals ~ temp + rel_humidity", data=df).fit()
y=mx+c -> y=m1x1+m2x2+.....+c  predict the number of rentals when the temp is 60F and the relative humidity is 50%, the point estimate is simply
y_pred = 52.9562 + 6.3829*60 - 2.7942*50  where coeff:
Intercept       52.9562      
temp             6.3829      
rel_humidity    -2.7942

y_pred is also called as the mean
What is the standard deviation of our residuals? Our best guess is the square root of the average squared residuals.
standard_deviation = np.sqrt(est.mse_resid)  mean squared error of residual, where est is the output of linear regression from 
above formula

We can guess the probability of atleast 400 rentals with 60F and 50% humidity using probability cdf:
print('Probability of at least 400 rentals at 60F with 50% humidity: '+str(1-norm.cdf(400,loc=296.22,scale=191.63)))  

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.3) -->split our data into a training set (70% of the data) and a test set (the remaining 30%)

SSR -> Sum of squared residuals/errors
SST -> Sum of squared total variation (variance) -> Baseline which is the mean to compare against is SST
Rsquare (coefficient of determination)-> SST-SSR/SST
OOS R-squared is out of sample rsquared focuses on its performance on unseen data
model we learned on the training data to predict rentals on the test data and then measure the OOS R^2
est_train = ols(formula="rentals ~ temp + rel_humidity + precipitation", data=df_train).fit()
test_pred = est_train.predict(df_test)
print('OOS R-squared: '+str(r2_score(df_test['rentals'],test_pred))) ->OOS R-squared: 0.29759583972864545

OVERFITTING:  it means it's become too focused on the specific details of the training data and has lost its ability to generalize to new, unseen data
------------ 
It deals with too many variables relative to size of dataset


IRRELEVANT INDEPENDENT VARIABLES or INSIGNIFICANT:
-------------------------------------------------
p value in ols summary, determines whether the independent variable is significant (<0.05) and not significant (>0.05). There is confidence interval as well
A low R² and low p-value (p-value ≤ 0.05) means that your model doesn't explain much of the variation of the data, but it is significant (better than not having a model).
A low R² and high p-value (p-value > 0.05) means that your model doesn't explain much of the variation in the data, and it is not significant (this is the worst-case scenario).
A high R² and low p-value means your model explains a lot of variation within the data and is significant (this is the best scenario).
A high R² and high p-value means that your model explains a lot of variation within the data, but is not significant (the model is essentially worthless).

confidence intervals next to pvalue also tells about significance. If the interval value contains 0, then its not significant. In our case 
rentals		0.000      -3.009      -2.578
random            	0.754     -11.745      16.206
In that, pvalue of rentals is <0.05 and the interval doesnt contain 0, for random, pvalue >0.05 and interval contains 0, hence not significant


IMPACT OF HIGHLY CORRELATED INDEPENDENT VARIABLES:
--------------------------------------------------
Remove highly correlated value from the list as it can lead to multicolinearity problem
In case of temp and temp_wb, both are correlated with 0.98. Anything greater than 0.75 try to remove one of them and then choose the variable which is correlated to dependent out of the 2. Reason being, it will impact the coefficients if you use both highly correlated independent variables.

TOO MANY INDEPENDENT VARIABLES:
------------------------------
less number of independent variables can lead to underfitting and too many can lead to overfitting. We need to make sure to have the correct number of independent variables where in-sample(training)/out-sample(test) Rsquare wont change much by adding more variables (called as optimal zone)

***In Summary, understand the context and start with variables you feel fits the model. Build Model and remove insignificant variables (p value >0.05), remove strongly correlated variables-  multi-collienarity (temp vs tempwb). Generate out of sample R-square and keep removing variables until you reach optimal zone.

****More variables is more model complexity
*** 0.2 as low correlation, 0.2-0.5 medium correlation, 0.5-0.8 high correlation, and above 0.8 as very highly correlated

PCA could decorrelate variables and reduce dimensionality

LOGISTIC REGRESSION:
--------------------
Regression is predicting something numerical - (count of bikes if weather is good, probability of loan getting default or not)
Classification is classifying into categories - (default a loan or not, spam filter, medical diagnosis, face recognition)
Generalized Linear Model(Logistic): Linear Model ->Sigmoid Function ->Probability (0-1)

Recommender Systems:
---------------------
Recommendation systems are algorithms that filter data toward the goals of predicting user preference. Based on that information, marketers and consumers can make decisions, for example, placing certain items next to each other in the grocery store. The goal is personalized content or a better product search experience. 
Netflix a subscription based online streaming and dvd rental service. Top picks, trending now, bcoz u watched narcos, these are your recommendations etc..
eharmony, ebay, amazon recommended systems are some examples
3 types:Collaborative filtering, Based on user-item interaction data used by netflix, amazon
	Content Based Filtering
	Matching Methods
Collaborative Filtering: User 2 archetypes weights: horizontal: W(jose,1), W(jose,2) etc..
			 Archetype 2 movies(Supposed rating for archetype): vertical: S(1,frozen), S(2,frozen) etc..
User's composition weights of archetype users using model fitting procedure: Rating of Movie(frozen) for Jose all archtypes: 									W(jose,1)*S(1,frozen)+W(jose,2)*S(2,frozen)...
Residual for model's predicted value for Jose for movie frozen: Residual(jose,frozen)= ObservedValue(jose,frozen) - W(jose,1)*S(1,fzn)+W(jose,2)*S(2,frzn)..
Least squares fit: square of residual   

Ensemble Learning:

Linear Optimization Models:
--------------------------
Optimization models ->Prescriptive Analytics
Decision variables: unknown variable to solve for, usually associated with decision. In buses, it is Xa, Xb, Xc = number of buses of type A, B, C
Objective Funciton: Goal to maximum avg fuel efficiency, 10Xa+8Xb+5Xc/Xa+Xb+Xc
Reqs/Conditions/Constraints: Cant spend more than allocated budget. 50Xa+70Xb <=10000; We have service req: 25Xa+50Xb+50Xc =20000 (students); Availability of driver: Xa+Xb+Xc<=450; Inventory of old bus: Xc<=400; Nonnegativity: Xa,Xb,Xc>=0

Linear vs Non Linear Function: 
------------------------------
9a+2b+3c+5 --> linear
9a+2bc+3ad+5 --> non-linear or even square of variable as well

Discrete Optimization:  (Binary and integer with linear functions) Integer Optimization and Non-linear optimization
----------------------
Linear optimization is variables can be of any real value (continous) within a specified range (including fractional values)
Linear Integer Optimization restricts Decision variables to integers (discrete). Solution must contain whole numbers rather fractional
Linear Binary Optimization restricts Decision variables to Binary (0 or 1). Solution must contain whole numbers rather fractional


Non Linear (Classification and Regression Trees CARTs) :
-------------------------------------------------------
**Linear regression is primarily used for predicting continuous numeric values. For example, predicting house prices based on features like area, number of bedrooms, and location.
**Logistic regression is used for binary classification tasks, where the target variable has only two possible outcomes (e.g., 0 or 1, True or False), commonly used for tasks like spam detection, medical diagnosis, and credit risk analysis.
Root Node --->Leaf/terminal nodes and depth(how much it traverse thro)
Gini coefficient is a way to quanity impurity and define binary classification problems (replacing MSE)
Summary: Take a variable and eash possible way to split the dataset with minimal reduction in mse/gini and split data 

Ensemble Learning:(free lunch learning):
---------------------------------------
Bagging: Bootstrap aggregation is like generating different training data sets, subtly different from each other using techique called bootstrap resampling (sample data with replacement)
Random forest: Do Bagging(creating multiple models) and do small tweak by randomly chosing few variables and every time you decide what variables to decide for generating the next split
Boosting: Build model look at the mistakes, build second model to minimize mistakes and so on by mitigating mistakes of previous ones (sequential approach). Boosting approach that uses trees is called gradient tree boosting
XGBoost: Adding penalty to gradient tree boosting. (after each round, minimize sum of sum of squared errors between current residuals and new model+penalty that favors small trees and small coeffs)
        Key default hyperparams are: max_dep=6, number of trees=100, learning_rate=0.3  there are other params but these are key ones where we need to play with numbers
	How to find good hyperparam combinations: 1. Manual trial and error 2. Automated hyperparam search (Grid search, random search)
Cross Validation:   It finds the best hyperparameter combination with the best error by comparing average errors derived from a number of k-fold cross-validation models. 

Ensemble Learning yields stronger out-of-sample predictive performance
Random Forest average together many independently constructed deep trees
Gradient tree boosting adaptively combines small trees
XGBoost incorporates regularization into gradient tree boosting and performs better in variety of problem settings
**problems involving structured data like tabular XGBoost is excellent choice
**problems involving unstructured data such as images, videos, nlp, deep neural networks is better choice

Tree Ensemble:
	Bagging (Bootstrap aggregating)
		Random Forest
	Boosting
		Adaptive Boosting
			Gradient Tree
		Gradient Boosting
			XGBoost (Xtreme Gradient Boosting), LightGBM (Light Gradient Boost Machine)

Fairness and Bias:
-----------------
Protected attributes: illegal to use distinguishing attributes of people to make decisions like: Age, gender, race, color, religion, national origin etc..
For instance: Illegal to use them in employment, in contract they can be used in healthcare service
Allocated Harm: Denies or grants certain opportunities to certain grps like loan application
Representational Harm: Diminishes the identity of individual or grp like wrongly identifying person for crime (false arrest)

Representation Bias: When data used to train model underrepresent or overrepresent certain groups. Ex: Conduct survey to understand preference of customer through online platform. Here you are excluding ppl who dont hve access to internet or not active online users.

Measurement Bias: Systematic error in way data is collected, measured, or recorded leading to inaccuracies in analysis, Using proxy labels.Ex: Cost expenditures to predict illness or lack of illness next year. Certain communities may not have access to healthcare. Ideally, need to hve patient's true state of illness (measurement)
MeasurementBias happens when you use convenient labels instead true lables.
Mitigate MeasurementBias:
------------------------
Work with domain experts to understand characteristics of true labels
Invest time and effort to get true labels
Example: Convenient Label: Classifying student performance as "poor," "average," or "excellent" based solely on standardized test scores.
True Label: Considering a broader range of factors such as class participation, extracurricular activities, and teacher assessments.
Mitigation: Engage with educators to understand the multifaceted nature of student performance. Gather data from various sources like teacher evaluations, peer reviews,

Mitigate RepresentationBias: 3 Broad approaches to Mitigate Representation Bias
-------------
Collect more data on underrepresented groups
Change inner workings of algorithms
Post process the model to satify some definitions of fairness
True Positive Rate: Equal opportunity across groups
Cut OFF RATE: It's a critical parameter in binary classification models as it determines the balance between true positives and false positives.
ROC curve: helps visualize the trade-off between TPR and FPR as the cut-off rate changes.Better predictive performance will have an ROC curve that is closer to top left

Neural Network: (AI->ML->DeepLearning) 
--------------
coefficients = weights
intercepts = biases
Execute Logistic regression by combining intercept (adding biases) + coefficient of variable 1 to n(multipying inputs with weights), then running through sigmoid function to determine probability 1/1+e^-x
**you transformed k-dimensional input into n dimensional vector output using linear function and then feed into nonlinear (like sigmoid). Ater n transformations, you then feed into logistic regresssion function
***use linear and non linear functions repeatedly between outputs and inputs so that there are transformations, to smartly represent unstructured data
***Neural network: Insert logistic or linear regression into linear functions, followed by non linear functions
***Neurons: operations that involve linear function and receive inputs, add them up and send them thro non-linear functions
One neuron connected to another neuron within a network of connections
**Layer: Vertical stack of neurons
**activation function: non linear functions (sigmoid) used inside each neuron including output
**input layer (inputs or variables with no transformations), output layer (final output from previuos of network procedures, sigmoid), rest is hidden layer
**dense or fully connected layer: layer with stack of neurons and numbers fed into every neuron in next layer
**deep neural network/deep learning: is a neural network with lots of hidden layers [number of layers is called depth of network] RESNET34
**Representations are build on previous ones and are hierarchial in nature

Activation Functions:
Sigmoid Activation Function:  1/1+e^-x (produces probability between 0 and 1)
Linear Activation Function: Function that takes input and passes it along unchanged
**Rectified Linear unit (ReLU): g(a)=max(0,a) [receives number, check if its positive then send unchanged, if negative then send 0]. It address vanishing gradient problem (if gradients become too small in large NN, then they fail provide meaningful updates to NN params leading to steady learning)

**Network Architecture: arrangement of neurons, activation functions and connections between layer

** Input Layer with 2 variables x1,x2 --> Hidden Layer [Neurons, lets say 3 (ReLU)] --> Ouput Layer 1 (Sigmoid)
Total weights and biases in the example are : 2*3=6 -> 1*3=3 -> 3*1=3 -> 1 ==Total 13 (9 weights and 4 biases)

NN2:
----
Output Varaible							Output Layer				   Loss Function
Single number(regression with single input) 		-->   Linear Activation function  -->		Mean squared error
Single probability (binary classification)  		-->   Sigmoid function	 	  -->		Binary cross entropy
Vector of n numbers (reg. with multiple o/p)		-->   Stack of linear activations -->		Mean squared error
Vector of n probs that add upto 1 (multiclass class)	-->   Softmax layer		  -->		Categorical cross entropy

Loss Function: Function that quantifies the error in a model's prediction and can be thought as error function (similar to Sum of squared errors in LR). Loss function tells the diff between predicted and actual. Lower the loss function, better the model. 

Gradient Descent: Start with a point, calculate derivative and then keep going until you find derivative that is closest to zero use a small value for alpha. New Point = current point - alpha*derivative    (repeat until derivative is closer to 0). Optimization alogrithm uses calculus to reduce loss function by calculating loss, then compute gradients of loss function by initially starting off with initial set of params (weights/bias) then using backpropagation to minimize loss function

Stochastic Gradient Descent: mini batch gradient descent makes it possible to work with extremely large datasets
Backpropagation: calculate gradient at output and update moving backwards from output to input (efficient organization of computation of gradient)

Regularization: technique to reduce overfitting/underfitting 
	Early stopping: stopping the training process when error on validation set flattens out or begins to increase
	Drop out: removing (dropping) a number of neurons randomly in each layer

Tensor: Tensor is a notion where single number like 6 has a tensor rank of 0, vector like series of numbers (43,3.5,34) has a vector rank of 1, a table with rows and columns has a tensor rank of 2, cube wit row,columns and depth has tensor rank of 3 and video clip containing frames, each frame being a cube, had 4 dimensions or 4 tensor rank

Tensor Flow: Its a library with numerous built in functions to manipulate and transform tensors. Automatic calcuation of gradient of complicated loss functions to minimize these loss functions. It provides state of art optimizers like SGD(stochastic Gradient descent) and its variants or siblings like ADAM

KERAS: Keras can harness all abilities of TF and provide more user-friendly concepts. Provides library of Activation functions, layers and flexible way to specify neural network architecture. It provides easy way to preprocess data, easy ways to train models and report metrics, easy access to pretrained models that u can download and customize

Classification Problems: 1. Classification (dog/cat) 2. Classification and Localizaton (dog/cat and where) 3. Object Classification and Localization
Grey Scale Pixel Range (0-255) where 0 is black and 255 is white
Color (RGB) Range (0-255) for all 3 matrices of numbers (channels)

Fashion MNIST dataset (grey scale image)

**Flattening data, we lost the local information about relationship of different aspects of image (like horizontal/edge etc.) thats when conv layer comes
Convolution Filter: its a filter applied to convolutional layer couldbe 2x2 or 3x3 etc..
Convolution Layer: is just a small square matrix of numbers
**Input image, you hover over the convolution filter(2x2 or 3x3 etc..) and multiply using RELU to create convolution layer-> This whole operation is called Convolution Operation

Feature Engineering: Computer scientists using human knowledge. Convolution filters are designed by hand
Pooling Layers:reduce size of output of intermediate layers. Pooling, such as max pooling, is a technique used in convolutional neural networks (CNNs) to downsample the spatial dimensions of feature maps generated by convolutional layers. It is primarily used to reduce the computational complexity of the model, decrease the number of parameters, and control overfitting.
Max Pooling Layer(empty block 2x2 etc): Super impose on top of convolution layer and take max of it and create remaining.
Convolution Block: is one or more convolutional layers(apply convolutional filter) followed by pool layer (max pooling layer)

***Input Image->1-n (Convolution Layers+Max Pooling Layer) aka Conv Block ->Flatten to long vector(RELU)->Output Layer

Kernel: A kernel (also known as a filter or a feature detector) is a small matrix applied to an input image to perform operations such as edge detection, blurring, or sharpening. 3x3, 5x5, 7x7 etc..
Stride: Stride refers to the number of pixels by which the kernel is moved across the input image each time during the convolution operation. Consider applying a 3x3 kernel with a stride of 2 to a 5x5 input image. The kernel will start at the top-left corner of the input image and move by 2 pixels horizontally and vertically for each step
Padding: Padding is the process of adding extra pixels around the borders of the input image to control the spatial dimensions of the output feature map.

Transfer Learning: Instead building deep network from stratch, we take a pretrained netwrok and customize to solve a problem
------------------
For Images type of data, you use Convolutional Networks, Sequences(audio, video, nlp) use recurrent neural network, transformers, for all->Residual
**publicly available pre-trained networks for image processing -> Inception, VGG, GoogLeNet, ResNet, EfficientNet, DenseNet
**pretrained deep neural networks -> https://keras.io/api/applications , https://www.tensorflow.org/hub  (publicly available datasets)
Networks Learns Smart Representation: 
	1. First Layer 	-> Detects lines, small circles, arcs etc.	
	2. Second Layer -> Detect complex edges such as honeycomb shapes, color gradients
	3. Final Layer	-> Detect complex forms such as animal shapes, human torsos

NLP: Classification of text(sentiment analysis), Prediction/generation of text(autocomplete), Information Retrieval(invoice, contracts and
---  understand if there is compliance), translation, content filtering, text classification, Summarization
Text Vectorization:  Standarize ->Tokenize ->Index ->Encode
  Standardize: As name suggest text in standard form like remove capitals,punctuation, accents, stop words(pronouns), perform stemming(go to root, like eat)
  Tokenization: Take input paragraph and splitting into tokens like "who","are","you"
  Index: Take each token and assign number
  Encode: Encode the indices using one hot enc (each token to unique vector), count enc(sum of vectors), multi-hot enc(same as count, non-zeros by 1

Word Encoding: Apply ratio of conditional probability to identify the probability of word in a sentence like Solid Ice has more than solid steam. Apply Vector representation of word and apply log transformation function to word vectors, you get number close to ratio of conditional probability.

GloVe (Global Vectors for Word Representation) is a word embedding technique that aims to capture the semantic meaning of words by assigning each word a high-dimensional vector, such that words with similar meanings are represented by vectors that are close together in the vector space.
GloVe is trained on corpus of text, where it creates a n-dimensional vector for a word using Co-occurence matrix(how often words appear together), Proability Ratioos, Training(stochastic gradient descent), vector space representation(predict to actual co-occurence probability adjustment), vector values(final values of vector representing the learned semantic features of words)

Transformer Architecture: (Developed in 2017)
------------------------
Accounts for surrounding context of each word, order of word, generates output that is same length as input
Attention layer: load the input embedding/words or vectors and then using attention layer with linear model produce the contextual info with no order
**Add the standalone embedding with position embedding and result embedding goes into the system
3 Reqs: Order, Context and Output same as input
Transformer Encoder: Takes order of words into account, Takes surrounding context of each word into accont and Gives as many outputs as it takes inputs
[Input embedding(each word has vector associated with it) + Positional Encoding] ->Input to Transformer [[Multi head attention, it transforms each of these encodings, so that encodings depend on context, which is order + Add and Norm (more of housekeeping)]+[Feed forward(add a hidden layer with ReLU)]]

Neural Network Summary:
======================
Model learns patterns and relationships with encoded arary using example lets say: "I really enjoyed the movie. It was fantastic"
Max tokens: this will get the most used and unique tokens in the entire dataset. 
Multihot encoding: Creates a binary enocoding of the words in every record/sentence and place according to its index, say for  "I" index 10, "enjoyed" index 100 etc..  [1,0,0,1,0,0,...] etcc
Models starts off with random weights and layers. During training, it adjusts the weights, in dense layer (ReLU), learns patterns by computing weighted sums of input features (binary array)
For example, one neuron in dense layer might learn to activate and detect presence of certain positive words input array and same with negative
Model learns statistical relationship between input feature(tokens) and output labels(sentiments) through backpropagation and gradient descent, model adjust weights based on errors in predicting
This iterative learning occurs over  multiple epochs until the loss function reduces with high accuracy. This optimized using Adam, loss function: binary/categorical cross entropy through output softmax/sigmoid layer 
Gradient descent is an optimization algorithm to minimze loss function by adjusting weights/biases (predicted vs actual in training data)
Backpropagation is process of computing gradient of loss function with respect to each parameter of NN. 2 Main steps Forward Pass and Backward Pass
Forward Pass, input data is fed thro network and predictions are made
Backward Pass, error between predicted and actual ouptut is computed using loss function
Stochastic Gradient Descent: instead using entire dataset unlike traditional one, uses a random sampel(small batch) from training data to compute gradients

Model Interpretability and Causality: (Understand/Explain how ML model makes predictions) - (ability for ML model to understand relationships within data)
====================================
Shapley values (effective in intrepreting black box models, but are compute heavy) allows to poke holes and peer into black box model
*Integrated Gradients: computationally less expensive. They are visual.

1. Generate linear interpolation(small steps in feature space) between baseline and original image. Generate interpolated images
2. Compute Gradients:  the gradient tells us which pixels have the strongest effect on the model's predicted class probabilities.
3. Accumulate Gradients: The integral_approximation(accumulate gradients) function takes the gradients of the predicted probability of the target class with respect to the interpolated images between the baseline and the original image.
4. Put all togther by utilizing tf.function decorator to compile into callable tensor graph.
Generate alphas, collect gradients, iterate alpha and batch computation and scale to large m_steps, accumulate. This process is called ig attribution by overlaying on original image

Intelligent PROCESS: Process (set of activities convert i/p 2 o/p), Actvities (algorithms, meetings etc), Resources(databases, tools etc), Output(actions)
==================   Decision (work) Process Metrics: Efficiency (output/input), Quality, Capacity(max throput), Cycletime(time elapsed), Flexibility
How we can leverage data, ai and ml to improve the process. 
**Process is a set of coordinated activities relying on various resources to transform inputs to outputs

Process Automation: Rule Based Automation->RPA (robotic process automation) -automate rule based task with finite set of rules to decisions in real timeML ML Enabled Automation: automate lot of unstructured data and produce a structured output (amazonGo,nauto AI dashboard etc.)
ML Enabled Monitoring and Sensing: Automated data collection and unification of data streams from different sensors. Algorithms to interpret the data and alert human experts to any potential failures(predictive maintenance like flight maintenance) - reduce downtime, eliminate unnecssary maintenance
Capabilities of Humans vs Machines:
Machines: have incomparable computational capacity and speed. It can interpret large and unstructured data. They can multitask and repetitive task. No Bias
Humans: can understand context, identify exceptions and changing conditions, detect nuances,  
How to decide: Consequences of errors, which decisions are more prone to errors, How complex/stable is the context compared to boudaries of model

***Personalization will result in -> Rules -> which will help in prediction of certain outcomes and take certain actions to certain segment of customers
Personalization: Sending coupons to and deals to people likely to purchase
Rule Based Automation: A digital Point of sale(POS) allow customers for order picks ahead of time and have them ready for pickup
ML enabled automation: Allows customers to see when their packages are going to be deliverd in real-time
ML enabled alert and sensing: allow airlines to understand which aircraft need maintenance beyond scheduled maintenance 

Three Lenses Framework for management success: when implementing intelligent process, we see workforce related issues. We use 3 lens strategies to implement change in holistic way 
=============================================
Strategic Design Lens: mechanical system crafted to achieve a defined goal, where parts much fit well and match demands [ex: technology, usefulness, correctness and workflow disruption concerns]
Political Lens: an organization is social system encompassing diverse and sometimes contradictory interests and goals [ex:Threats to user identity including experience and expertise concerns]
Cultural Lens: an organization is a symbolic system of meanings, artifacts, values and routines [ex: medical professionals interacting with data predictions provided by IT]
How to mitigate: using feedback to improve tool, adapting model's output, implement workflow process appropiate for tool. Start with centralized implementation
Example: school "healthy lunch" program started by elementary school mgmnt (power aka political lens) without coordinating. Not understanding the culture like what kind kids like, picky eaters etc(Cultural lens) and not getting feedback from the parents (strategic design)

Leading Transformations:
========================
The Playbook: 
	Phase 1: Problem definition, understanding system, data acquisition [establish data driven approach to identify n validate hypothesis]
			failures: people start project with solution already and validate it with data , people rush, 
	Phase 2: Data-driven modeling [use hypothesis testing to explore impact of changes and interventions]
			failures: people prescribe a single solution, models should present pareto curve
	Phase 3: Implementation and dissemniation [develop tools with SMEs and DSs and expand scope and exposure of tools to stakeholders]
			failures: lot of politicial and culture challenges 
	Phase 4: Monitor and improve [monitor outcomes and impact of project, important to identify changes in system that require change in solution]
			failures: mask the true impact, failure to measure regularly will result in evaporation 

Business Translators: understand/measure biz metrics and goals, translate them to questions that can be answered by data -> concrete metrics that can be measured[more qualitative, data driven] -> help DS to understand biz constraints and ensure these contraints are captured in their models [bridge data world to business world]


Fine tuning: with training corpus
self supervised -> Filling the next word
supervised -> Q&A , send the 
Reinforcement Learning -> promtt->supervised FT model, send completion to Reward model and send feedback to SFT and so

1. Choose fine tuning task -> text summarization, text generation, classification etc..
2. Prepare training dataset -> i/p , o/p pairs and desired summarization and generate training corpus
3. Choose Base model 
4. Fine tune model using supervised learning
5. Evaluate model performance
