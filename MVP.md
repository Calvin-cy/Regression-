# NBA regression analysis

After web scraping from Espn and basketball-reference.com, I obtained 2833 columns along with 31 features. I selected the useful features and created this pairplot, and realized that fitting a simple linear regression model may not work because the scatter plot shows no sign of pure linear relationship


![pairplot](https://user-images.githubusercontent.com/63031028/114510943-22996f00-9bec-11eb-85fa-1125c190967c.png)

- Linear Regression val R^2: -337707308563977600.000
- Degree 2 polynomial regression val R^2: 
- Ridge Regression val R^2: 0.507-4540475800458.793
- Lasso Regression val R^2: 0.506

I tried different regression model and found out that Ridge and Lasso work for my dataset. 


The MVP is to keep digging into Ridge and Lasso, create a plot to find the best alpha of each one, and add or reduce some features to get the best score of R-squared
