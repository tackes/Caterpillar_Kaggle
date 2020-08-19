# Caterpillar_Kaggle
R Code for Caterpillar forecasting Competition on Kaggle

INTRODUCTION:
Background on Caterpillar
Caterpillar is the largest construction equipment manufacturer in the world. In 2017, the big yellow company grossed over $45.5 billion dollars.  CAT has a variety of  huge construction and mining equipment stationed around the world.  Founded in 1925, Caterpillar is an American Fortune 100 corporation.  Their origins can be traced to the merger of the Holt Manufacturing Company and the C.L Best Tractor company, creating the Caterpillar Tractor company.  In 1986, they rebranded under the current name, Caterpillar, Inc.
.  
BACKGROUND:
Purpose of the Competition
These mechanical beasts rely on several complex tubing networks to ensure the equipments do their job properly.  Each tube has their own set of unique properties and characteristics.  The tubes differ in dimensions, materials, number of bends, bend radius, bolt patterns, and end types.  As the cost of tubes and being able to forecast tube cost of tubes is crucial to CAT’s ability to forecast future build costs, they started this competition to try and predict the price a supplier will quote for a given tube assembly.
Competition
The Caterpillar Tube Pricing Kaggle competition started and ended in August of 2015.  The competition awarded $15,000 to the top placing team, with a second and third place price of $10,000 and $5,000 respectively.  Participants are allowed to submit up to 5 results a day, which are graded using 30% of the test data.  The standings for these results were posted to a “public” leaderboard.  At the end of the competition, participants were then asked to submit their best 2 results for final submission, which would then be graded using 70% of the test data and used in the final standing, with the results being posted to a “private” leaderboard.  The final evaluation metric is the Root Mean Squared Logarithmic Error (RMSLE).
 
The RMSLE measures the ratio between the actual and the predicted value.  It can be used as a metric when you do not want to penalize big differences when both values are large.  In addition, it tends to penalize under estimates more than over estimates (Oliveira, et al., 2017).
EXPLORATORY DATA ANALYSIS:
A total of 21 files were provided for analysis and modeling.  Each file contained important aspects of tube assemblies ranging from quantities purchased, features of tube assemblies and the price paid for the products.  Some of the data sets could be linked using tube assembly ids, while other data sets used component ids.  All files were csv files. 
  
Train and Test Sets: A train set and test set contained price quotes from suppliers, the quantity ordered, the supplier id as well as minimum purchase quantities and whether bracket pricing was available.  Both files contained tube assembly ids as a unique identifier.  The price paid was removed in the test data, which was the target of the competition. 
Since test and train sets contained a tube assembly id, the files were merged together as well as with the other files: a.) bill of materials, b.) specs, and c.) tube.  The variables appearing in the files are listed in table 1.

Component files: A total of 12 files contained information about various aspects of the components used in tube assemblies. Each file had a component_id, which was used as a unique identifier during the merging of these files. The component id also appeared in the bill of materials file, which made it possible to merge the 12 component files and the assembled data set containing training and test data.  See table 2 for full table structure. Four additional files were provided to name component types, connections, end forms and tube end forms.  See table 3 for full structure and variables.

Since cost was the target variable, it was plotted against several variables describing tube assemblies (e.g. diameter, length, wall and bend radius). Four examples are provided as scatter plots (See figure 1).  Minimum order quantity did not have an influence on price, although some interesting observations in the extreme range of above 1,000 show minimal prices.  This means that there were some very large order quantities for which Caterpillar was charged a substantially lower price.  Other variables depicting features of tube assemblies had minimal relationship with price as well as depicted by the near flat regression line..  

The average tube price per volume order did have an inverse relationship with price: as the volume order increased, the cost seemed to decrease.  
DATA PREPARATION:
Feature Engineering
Despite having over 1000 variables after merging all our tables, feature engineering was critical to a successful model. We added several variables to try and pull out more prediction power from our models. 
Dates
In our original dataset, we were provided a standard quote date in mm/dd/yyyy format.  From that, we extracted:
●	Quote Year
●	Quote Month (1-12)
●	Quote Day (1-31)
●	Quote Day of week (0-6 with Sunday being 0)
●	Quote Week of Year (0-52)
We added 1 to the day of week and week of year to change our values from 1-7 and 1-53 respectively.  This added significant prediction power to our model, with quote year being the second most important feature in our champion model.  

An investigation of relationship between cost and the timing of orders revealed interesting relationships (See figure 3).  For example, orders between 2005-2010  could be more costly than orders in other years.  Some orders had a higher cost in the middle of the year, and the quote day of the month also may have influence the cost.  As a result, the  time features were considered be good predictors in a model, worthy of further investigation. 
Aggregate Numerical Data
The structure of the data led to several repeated variables for the many different subassemblies inside the tubes.  Some tubes for instance, had up to 8 sub-components.  Each of those 8 sub-components then had 65 unique characteristics of their own. Our goal was to try and aggregate all the like variables into min, max, mean, and sum.  In total, there were 39 variables, each of which we used to create 4 new aggregate variables.  Table 4 shows all the numerical variables that were aggregated.

These new variables played a key role to increasing the predictive power of our model.  Some of the variables were obvious such as quantity.  It is logical that as the quoted quantity changed, so did the quote price.  Some of the less obvious important variables were length and number of bends, potentially indicating the impact on cost as the size and complexity of the tube increased.

Transformations
The only transformation we applied to our model was on the dependent variable, cost.  In many regression problems, “it is often advantageous to apply a transformation to a response variable y.” (Huang & Draper, 2003).  As figure 4 shows, the original histogram is clearly not normally distributed. In fact, it is extremely skewed right.  To attempt to normalize our cost variable, we took a log(cost+1) transformation.  The plus 1 is to ensure there are no 0 quotes that will mess with the distribution. The result is a much more evenly distributed variable more closely resembling a normal distribution.  The goal of this transformation is not to make our graph simply look better, but rather to make the relationship between the variables more linear.
Preparation for Analysis
MODEL BUILDING:
Several models were built: a.) Random Forests, b.) Boosting, and c.) Extreme Gradient Boosting.  After feature engineering and variable selection, all variables were used when building models.  All models were trained on the training data set, and deployed on the test dataset without the target variable, e.g. “cost”.  The predictions then were submitted to Kaggle for scoring. 

Random Forests: Decision trees are built on bootstrapped training samples during the creation of random forests. Since random forests force splits to consider a subset of predictors, the trees become decorrelated and the model becomes less variable, according to James, Witten, hastie and Tibshirani (2015).  Unfortunately, the process of building random forests on a relatively large dataset is a slow process due to increased computation time.  The problem was resolved by using the h2o package, which allows a more efficient CPU capacity utilization.    
The random forest package uses p/3 variables when using regression trees by default (mtry parameter in randomForest package).  Since a very large number of predictors were used, the mtry parameter was left as default.  The number of trees grown was specified at 100.  

XGBoost: Boosting requires a sequential assembly of decision trees using bootstrap sampling.  One of the advantages of using XGBoost is that the development of models took significantly less computing time vs. boosting (Chen and Guestrin, 2016).  In fact, Jain (2016)  listed seven advantages of XGBoost over traditional gradient boosting, six of which were extremely relevant in the Caterpillar competition (Srivastava, 2016):
1.)	Overfitting is less likely in XG Boost because the method is regularized.  Regularization uses a penalty when more features are used in a model. 
2.)	Despite the sequential nature of boosting, XB boost uses all cores of a computer during processing, according to Jain (2016). This was one of the reasons for faster computing times vs. competing models. 
3.)	XGBoost provides the option of tuning parameters including general parameters and booster parameters.  Learning task parameters can also be changed to fit a particular  problem.  
4.)	The Caterpillar data had a large amount of missing data, which XGBoost handled well. 
5.)	Boosting in general is a greedy algorithm, but the method provides the ability to specify the maximum depth for trees, and then prune backwards, and remove unacceptable splits (e.g. splits with no positive gains), according to Jain.  
6.)	Cross-validation was used on the training data to tune parameters, which is a built in function 

Following the recommendations of Srivastava (2017), a general parameter called booster was set to gbtree and the objective function was specified as reg:linear.  Experienced data scientists suggested that a tree booster should be used because it frequently obtains superior results vs. a linear booster. 
As a second step, booster parameters were specified.  Eta specified the learning rate.  While the default of eta is 0.3, the Caterpillar model’s eta was specified at 0.005. 
In order to better control overfitting, the sum of weights was set at 5 (the parameter is called min_child_weight).  Using cross validation to find the the sum of weights was important to avoid over or underfitting.  
The maximum depth was set at 6.  Again, this parameter controls overfitting.  Analysts suggest to set this parameter between 3-10 (Jain, 2016).  
The maximum number of features was determined by the parameter colsample_bytree, which was set at 0.6 
Feature Importance
With over 1000 predictor variables available for XGBoost to choose from, much of the predictive power resided in the top 20 variables, with a considerable drop off after the first 4.   Feature importance is broken down into 4 main areas:
1.	Cover
2.	Gain
3.	Frequency
4.	Importance
Cover metric is the relative number of observations related to this feature. In other words, how many observations used this particular feature to decide the leaf node relative to all other feature cover metrics.
Gain implies the relative contribution of the predictor calculated by taking each feature contribution for each tree in the model.  A higher number means more contribution.
Frequency is the percentage representing the relative number of times a given feature occurs in the various trees in the model, relative to all other frequency weights.
Gain is the most relevant attribute in determining the importance of each feature, and is also why our gain value and importance values are all the same.
Our first few top performing variables intuitively make a lot of sense.  Suppliers each have their own pricing structure and methodology, which is why it was our most important predictor.  Quote year being our second most powerful predictor, shows that pricing changes over time. Annual usage and quantity add a demand factor into the equation, which will affect pricing as the supply and demand of that tube and its subcomponents change.  Length, number of bends and volume add in a component complexity factor into the model.  Our top 10 predictors are shown in figures 5-8 with the top 20 shown in table 1. 
In preparation of boosting, each feature was clustered using k-means clustering.  Models were independently (and automatically) built for each cluster.  Since the target was the same, these were sub-models of the same underlying parent model. The visuals clearly show the most important features, which can then be easily observed in the table as the gain or importance columns. 
                                                                                                                                                                                                                                           
RESULTS:
Random Forest	
Random Forest and XGBoost were the main 2 models we were focused on.  
In Random Forest, each tree is trained on ⅔ of the training data.  Cases are drawn at random with replacement.  The remaining ⅓ of the data is used for testing the model fit.  Our model results were respectable using a random forest model.  The RF model scores a 0.2162 in the root mean square error (RMSE) metric. One benefit of the RMSE is that is has the same units as the dependent variable. It measures the standard deviation of the unexplained variance.  It is a good measure of how accurately the model predicts the response variable.

MSE:  0.04676378
RMSE:  0.2162493
MAE:  0.1057102
RMSLE:  0.06260415
Mean Residual Deviance :  0.04676378

With high hopes for this model, after submission our RF model with 2000 trees scores a 0.246719 on 70% of the data using a room mean squared logarithmic error (RMSLE) metric. 
XGBoost
XGBoost is a leading model when working work tabular data and often dominates many Kaggle competitions.  Modeling 1000 rounds using an eta parameters of 0.05, our 3-fold cross validated RMSE was 0.192041, significantly higher than our best random forest model.

[999]	train-rmse:0.091148+0.000202	test-rmse:0.192053+0.002091 
[1000]	train-rmse:0.091091+0.000227	test-rmse:0.192041+0.002093

After numerous tuning attempts, our final submission used 20,000 rounds and an eta of 0.005. The submission score using this model was our top performer and resulted in a RMSLE score of 0.219270 based on 70% of the data.   Had this competition been active, our placing on the leaderboard would have been 116 / 1323, of the top 8.7%.  Figure 8 and 9 show our top performing model compared to where we would place on the leaderboard during an active competition.
Ensemble
We did attempt ensemble approaches with the random forest and xgboost models, however while improving the score of our random forest model, the ensemble models brought down our standalone xgboost model.
DISCUSSION:
Feature generation turned out to be beneficial when creating models.  In fact, the most important predictors were all generated variables from original features.  Two of the most important predictors (maximum and minimum quantity) were at least three times more influential in determining the price than the third and fourth predictors (e.g. annual usage and quantity), which in turn were significantly more influential than any other variable.  Note that the maximum   and minimum quantity ordered were far more influential than the actual quantity provided by the Caterpillar data set.
Other important (albeit less influential) features were various depictions of product weight in addition to the identity of the supplier.  In terms of influence, other predictors were significantly less influential.  As an example, the gain of the first feature (max. quantity) was 0.21 compared to the fifth most influential feature (mean weight) with a gain of 0.04.  In order to better illustrate the rapid decline in gain, the 10th feature (wall volume) had a gain of 0.03, while the 20th most influential feature’s (component ID 1) gain was only 0.008.  Note that there were hundreds of features with extremely low gain scores. 
It is worth noting that XG Boost outperformed our random forest model. This is not surprising because XG Boost uses a slightly different algorithm to learn tree structures, according to Synced (2017).  XG Boost uses a Hessian matrix, which has an impact on the tree structure and leaf weights.  The Hessian matrix has a higher order approximation then random forests, which results in a better tree structure. 
The author also argued that XG Boost wins almost every competition because boosting determines local neighborhoods, and the method is less susceptible to the curse of dimensionality.  Further, Srivastava (2017) argued that the method is more efficient and provides more accurate predictions than other tree-based models due to its “slow implementation” .  

CONCLUSION:
XG Boost is a powerful technique that is useful to predict continuous as well as categorical target variables.  The technique allows the tuning of parameters that often results in significant gains in predictive accuracy when the model is deployed on test data.
Our XG Boost model outperformed our Random Forest model   as well as ensemble models that included both XG Boost and Random Forest components.  As a result of tuning some of the parameters such as min_child_weight, eta, gamma, subsample, max_depth, etc,and resulted in a  the test accuracy of the model vastly improved reaching an RMSLE score of 0.219270.  While the competition was no longer active, our RMSLE score would have ranked us as 116 of 1,323. 

LIMITATIONS:
The Caterpillar data was provided to an audience with no knowledge of the heavy machinery industry.  Further, the cost of tube assemblies had to be predicted by using a set of variables.  Caterpillar may have other information and knowledge not provided in the competition, making the predictive accuracy of models built by employees of Caterpillar potentially superior.   
