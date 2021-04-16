# NBA Salary Predtiction : Linear Regression model and Web Scraping 
Calvin Yu
## Abstract
The goal of this project is to predict the salary of an NBA player given his stats. The first step is to perform a web scraping task on Basketball-reference.com and espn.com to get the data that is needed for this project. After acquiring the data, I clean up the data and adjust them into a readable format. The data has so many features, and it is hard to fit all regression models. So I decided to select the features which I think will be important factors in predicting salary. I separate the data into three groups: training, validation, and testing. I have selected several models and train them with the training data to see which one can provide the best explanation.

## Design
The NBA nowadays has shifted from the low-scoring, defense-first, low pace, Center-based era, to a higher scoring, offense-first, high pace, and Guards-based era. As a NBA fan, I am interested in how the stats of a player contributed to his salary. Therefore, I would like to explore the question: How much will a player get based on his contribution to the team, or his 'numbers'.

To answer this question, I will create a machine learning algorithm that takes the stats from each player in 2015 to 2021 and predict their salary. 

## Data
### Basketball-reference (Web-scraped)
- The seasons of 2014/2015 to 2020/2021
- Player per game stats
- Combined each season to a big data set with 2833 data points with the average for the nba players who participated during these six season and 29 feature columns. 
- feature highlights: Points, Assists, Rebounds, Minute plays, and efficiency Field Goal % 


### ESPN (Web-scraped)
- The seasons of 2014/2015 to 2020/2021
- Salary from each player 
- 2833 data points * 2 features (Player Name, Salary)

## Algorithms 
1. Use the Requests and Beautiful soup to download the html pages for both basketball-reference.com and espn.com
- ### For basketball-reference.com 
2. Use the soup.find method to obtain the columns and
 Use soup.find_all method to find all useful text under the columns and append them into a list
 3. Use the pd.DataFrame method to restructure the list with the columns name 
 4. Repeat step 2 and step 3 for until we get all the data for 2014/2015 to 2020/2021

Example :
```
 head2021 = soup2021.find(class_='thead')
 column_names = head2021.text.replace('\n',',').split(',')
 column_names = column_names[2:-1]
 table2021 = soup2021.find_all(class_ = 'full_table')
 players = []

for i in range(len(table2021)):
    
    player_ = []
    
    for td in table2021[i].find_all('td'):
        player_.append(td.text)

    players.append(player_)
    df2021 = pd.DataFrame(players,columns = column_names).set_index('Player')
```

 ### For espn.com
 5. Create a for-loop to extract all the pages into a readable html page
 6. use the find_all method to find the rows we need, and create a list to append them 

 Example :
 ```
 salary2021 = []

for i in range(1,14):
    url = "http://www.espn.com/nba/salaries/_/year/2021/page/{}".format(i)
    r = requests.get(url)
    soup = BeautifulSoup(r.content,'lxml')
    
    odd = soup.find_all("tr", {"class": "oddrow"})
    for item in odd:
        table_odd = item.text
        salary2021.append(table_odd)
    even = soup.find_all('tr',{'class':'evenrow'})
    for item in even:
        table_even = item.text
        salary2021.append(table_even)
```
7. export the salary data into a csv file and do the split the columns on excel 
8. After the splitting is done, read the csv file that we created by using pd.read_csv and do the final editing 
9. Merge the basketball-reference dataframe with the salary data frame and concatenate all the years into a big dataframe and I add a year column on the big dataframe
10. Since most of the features on our big data frame are object type, we use pd.to_numeric to change the features that need to be evaluated numerically 
11. Multiply the eFG% and FT% column by a hundred since the regression model doesn't do well on numbers between zero and one 
12. Some of the players can play multiple position, and the data set categorize them with both the positions: 'C-PF , G , F'. Therefore, I will clean this column by grouping them into their true position
13. Some of the features contain Nan values, for example, some of the players have never shot any free throw and therefore will have a Nan value on the FT% column. I fill the Nan value with the mean of all the NBA player % to avoid dealing with outliers 
14. Explore the outliers and reassign them with a proper number 
For example, there is one player who played 83 games in 2015, and I believed it is a typo because there were only 82 games in the 2015 season 
15. Since there are so many features, I chose the ones that I think will be the most useful to perform the machine learning algorithm since too many features may have caused Multicollinearity 
16. Create pairplot to see how each features correlated with each other 

![PP](https://user-images.githubusercontent.com/63031028/114988199-e9126f00-9e4a-11eb-8a6c-0b10afd5b96f.png)

17. Using Linear Regression Model, Lasso ,Ridge, Polynomial(degree = 2) , Lasso of Polynomial(degree = 2) , and Ridge of Polynomial(degree = 2) to fit the best model 

18. Create a function to find the best alpha for Lasso and Ridge:
```
def build_grid_search_est(model, X, y, cv=5, **params):

    kf = KFold()
    grid_est = GridSearchCV(model, param_grid=params, cv=kf, 
                            return_train_score=False)
    grid_est.fit(X, y)
    df = pd.DataFrame(grid_est.cv_results_)
    for param in params:
        df[param] = df.params.apply(lambda val: val[param])
    return grid_est
```
19. Using a function to separate the data frame into 3 sets: training, validation, and testing to see the score of each model, also used the kFold method make the model more accurate :
```
def split_and_validate(X,y):
    
    X, X_test, y, y_test = train_test_split(X, y, test_size=.2)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25)
    
    kf = KFold(n_splits=10, shuffle=True)
    
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    lm_cross_val_score = cross_val_score(lm, X_train, y_train, cv=kf, scoring='r2')
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)
    X_test_scaled = scaler.transform(X_test.values)
    

    ridge_grid_est = build_grid_search_est(Ridge(), X_train, y_train, cv=kf,
                                alpha=np.logspace(-4, -1, 10))
    reg_alpha = ridge_grid_est.best_estimator_.alpha
    
    
    lm_reg = Ridge(alpha = reg_alpha)
    lm_reg.fit(X_train_scaled, y_train)
    lm_reg_cross_val_score = cross_val_score(lm_reg, X_train_scaled, y_train, cv=kf, scoring='r2')
    ridge_grid_est = build_grid_search_est(Ridge(), X_train, y_train, cv=kf,
                                alpha=np.logspace(-4, -1, 10))
    
    
    
    lasso_grid_est = build_grid_search_est(Lasso(), X_train, y_train, cv=kf,
                                    alpha=np.logspace(-4, -1, 30))
    lasso_alpha = lasso_grid_est.best_estimator_.alpha
    lm_lasso = Lasso(alpha = lasso_alpha)
    lm_lasso.fit(X_train_scaled, y_train)
    lm_lasso_cross_val_score = cross_val_score(lm_lasso, X_train_scaled, y_train, cv=kf, scoring='r2')

    
    poly = PolynomialFeatures(degree=2) 

    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    X_test_poly = poly.transform(X_test)
    
    
    
    X_train_poly_scaled = scaler.fit_transform(X_train_poly)
    X_val_poly_scaled = scaler.fit_transform(X_val_poly)
    X_test_poly_scaled = scaler.fit_transform(X_test_poly)

    lm_poly = LinearRegression()
    lm_poly.fit(X_train_poly, y_train)
    lm_poly_cross_val_score = cross_val_score(lm_poly, X_train_poly, y_train, cv=kf, scoring='r2')
    

    ridge_grid_est_poly = build_grid_search_est(Ridge(), X_train_poly_scaled, y_train, cv=kf,
                                    alpha=np.logspace(-4, -1, 10))
    ridge_alpha_poly = ridge_grid_est_poly.best_estimator_.alpha
    lm_reg_poly = Ridge(alpha=ridge_alpha_poly)
    lm_reg_poly.fit(X_train_poly_scaled, y_train)
    lm_reg_poly_cross_val_score = cross_val_score(lm_reg_poly, X_train_poly_scaled, y_train, cv=kf, scoring='r2')
    
    
    lasso_grid_est_poly = build_grid_search_est(Lasso(), X_train_poly_scaled, y_train, cv=kf,
                                    alpha=np.logspace(-4, -1, 30))
    lasso_alpha_poly = lasso_grid_est_poly.best_estimator_.alpha
    lm_lasso_poly = Lasso(alpha = lasso_alpha_poly)
    lm_lasso_poly.fit(X_train_poly_scaled, y_train)
    lm_lasso_poly_cross_val_score = cross_val_score(lm_lasso_poly, X_train_poly_scaled, y_train, cv=kf, scoring='r2')
    
    print(f'Linear Regression val R^2: {lm.score(X_val, y_val):.3f}')
    print(f'Linear Regression mean val R^2: {np.mean(lm_reg_cross_val_score):.3f}:')

    
    print(f'Ridge Alpha Best_estimator : {reg_alpha:.3f}')
    print(f'Ridge Regression val R^2: {lm_reg.score(X_val_scaled, y_val):.3f}')
    print(f'Ridge Regression mean val R^2: {np.mean(lm_reg_cross_val_score):.3f}')
   
          
    print(f'Lasso Alpha Best_estimator : {lasso_alpha:.3f}')
    print(f'Lasso Regression val R^2: {lm_lasso.score(X_val_scaled, y_val):.3f}')
    print(f'Lasso Regression mean val R^2: {np.mean(lm_lasso_cross_val_score):.3f}')
    
    print(f'Degree 2 polynomial regression val R^2: {lm_poly.score(X_val_poly, y_val):.3f}')
    print(f'Degree 2 polynomial Regression mean val R^2: {np.mean(lm_poly_cross_val_score):.3f}')

    
    print(f'Degree 2 polynomial Ridge Alpha Best_estimator : {ridge_alpha_poly:.3f}')      
    print(f'Degree 2 polynomial Ridge Regression val R^2: {lm_reg_poly.score(X_val_poly_scaled, y_val):.3f}')
    print(f'Degree 2 polynomial Ridge Regression mean val R^2: {np.mean(lm_reg_poly_cross_val_score):.3f}')

          
          
    print(f'Degree 2 polynomial Lasso Alpha Best_estimator : {lasso_alpha_poly:.3f}')  
    print(f'Degree 2 polynomial Lasso Regression val R^2: {lm_lasso_poly.score(X_val_poly_scaled, y_val):.3f}')
    print(f'Degree 2 polynomial Lasso Regression mean val R^2: {np.mean(lm_lasso_poly_cross_val_score):.3f}')
```
20. Doing some featuring engineering and test the result with the split_and_validate and function 
21. After playing with the features, the best dataset is adding the year as a categorical column and create a dummy variable 
22. The best model is the Ridge of Polynomial(degree = 2)
23. The score for the training data set is 0.71 and the validation set is 0.64, and the testing set is 0.64 as well.
24. Since we use polynomial regression, we need to unscale the coefficients and the intercept (Please see the code for the unscaled coefficients as there are so many of them to list out)
25. Visualize actual vs prediction:

![avp](https://user-images.githubusercontent.com/63031028/114993715-f03c7b80-9e50-11eb-8a63-7940eb68119d.png)
## Tools
- Requests - requests the URL
- BeautifulSoup - interpret it to python readable format
- Pandas - data manipulation and data aggregation
- Numpy - data cleaning 
- Seaborn - visualization
- Matplotlib - visualization
- sklearn - machine learning algorithms: create different models and find the best R^2 score of each model 


## Communication
- The best model can explain 64% of all the variability  of the target: Salary

- The plot of actual vs prediction shows that
many points fall under the dotted line as the Salary goes up. This indicates that this model will tend to over-predict the salary based on our features. 

## Further steps
Here are the following that we could explore more in the future: 
1. Download more data from the basketball-reference.com because they have multiple data sets like stats per 36 minutes, stats per 100 Possession, and advanced stats,.etc, that can probably feed our algorithm more features for it to increase its R^2 score 
2. Explore the NBA players belonging teams to find out if their teams are on a rebuild, or chasing for the championship

## Takeaways
1. How to perform a web scraping task 
2. How to split the data to train/validation/test and train them with different models
3. The importance of R^2 

