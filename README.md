
#  [Walmart Recruiting : Sales in Stormy Weather (Regression Analysis)](https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather)

<img src="https://github.com/lucaseo/dss7-walmart-project/blob/master/archive/final/Presentation_fin/data/walmart.jpg">

#### [Click for Project Report](https://github.com/lucaseo/dss7-walmart-project/blob/master/project_report.ipynb)


# [ Overview ]

### (1) Dataset : 
> #### Walmart Sales Records and Corresponding Weather Records

### (2) Objective :
> #### Trip Type Classification of each customers based on thier shopping data

<br>

### (3) Dataset Description : 
> #### train : 4617600 rows, 4 columns  
> - Sales record of 111 items throughout 45 Walmart between 1 JAN 2012 ~ 31 DEC 2014

> #### key : 45 rows, 2 columns
> - Index of weather Stations in 20 different locations, and 45 Walmart which shares the same region with the weather stations.

> #### weather : 20517 rows, 20 columns
> - Weather data of each weather station between 1 JAN 2012  ~ 31 OCT 2014. (Specific descriptions on weather columns can be found in project report)

> #### test : 526917 rows, 3 columns
> - Sales count prediction template of 111 items throughout 44 Walmart between 1 ARR 2013 ~ 26 DEC 2014 

### (3) : Evaluation
> #### Root-Mean-Squared-Logarithmic-Error(RMSLE)  
> <img src="https://github.com/lucaseo/dss7-walmart-project/blob/master/archive/rmsle_metrics.png">  
> n is the number of rows in the test set  

> p is your predicted units sold  

> a is the actual units sold  

> log(x) is the natural logarithm  




# [Contents]

### (1) EDA 
> - Tendency Check
> - Missing Data

### (2) Preprocessing
> - Various Missing Weather Data Inputation approaches
>   - by mean
>   - by z-score of closest kneighboring data
>   - by refering to previous day weather + filling out weather data by meteorological formulas
> - Parsing datas
> - Encoding of categorial variables 
>   - codesum
>   - weekday, weekend

### (3) Modeling
> - OLS with weather features (1)
>   - Removed outliers
>   - Removed features with high multicolinearity based on Variance Inflation Factor
> - OLS without weather features (2)
> - Apply log on target variable

> - R-squared : 0.90

### (4) Prediction

### (5) Kaggle Submission
> - Total Teams : 485 teams
> - Final Score : 0.19943 (RMSLE)
> - Leaderboard : 317 / 486


## What could've been better : Lesson learned
- EDA focusing on how features affect target variables helps gaining insights of feature engineering while modeling.
- Log(x+1) on target variable instead of log(x). Needs better understanding of evaluation metrics.
- Giving interaction between features could generate coefficient that better explains the target variable.
- Weather data is based on timeseries --> Time series analysis could help find relationship between weather and item sales.
- Additional information of US federal, national holiday and seasonal shopping event.
