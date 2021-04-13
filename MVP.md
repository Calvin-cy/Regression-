# NBA regression analysis

After web scraping from Espn and basketball-reference.com, I obtained 2833 columns along with 31 features. 


![possalary](https://user-images.githubusercontent.com/63031028/114515940-c0dc0380-9bf1-11eb-989e-c56496ffe62e.png)

For example, I used Age as a feature and Position as a hue to see if there is a relationship between salary and age. This graph tells me that there is, and experienced Point Guard benefits the most in aging. In order to predict how the features contributed to the growth of salaries, I tested all the features with four separate models.

- Linear Regression val R^2: -337707308563977600.000
- Degree 2 polynomial regression val R^2: -4540475800458.793
- Ridge Regression val R^2: 0.507
- Lasso Regression val R^2: 0.506

I found out that Ridge and Lasso work for my dataset. 


The MVP is to keep digging into Ridge and Lasso, create a plot to find the best alpha of each one, and add or reduce some features to get the best score of R-squared
