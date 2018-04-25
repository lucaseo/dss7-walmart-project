##### Fast Campus Data Science School 7th Team Project 2 Classification
#  [Walmart Recruiting : Sales in Stormy Weather (Regression Analysis)](https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather)

<img src="https://github.com/lucaseo/dss7-walmart-project/blob/master/archive/final/Presentation_fin/data/walmart.jpg">

#### [Click for Project Report](https://github.com/lucaseo/dss7-walmart-project/blob/master/project_report.ipynb)

### Team : Tetris
> - [Lee JW](https://github.com/anylee)
> - [Seo WY](https://github.com/lucaseo)
> - [Kim DH](https://github.com/)

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



# [Contents]

### (1) EDA 
> - Tendency Check
> - Missing Data

### (2) Preprocessing
> - Missing Data Inputation
> - Encoding

### (3) Modeling
> - OLS with weather features
> - OLS without weather features
> - R-squared : 0.9079

### (4) Prediction

### (5) Kaggle Submission
> - Total Teams : 485 teams
> - Final Score : 0.19943 (RMSLE)
> - Leaderboard : 317 / 486