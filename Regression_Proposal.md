# Regression Proposal

## Question/need:
- To predict the Fantasy score for each NBA player in their upcoming game 
- Players from different platform of NBA fantasy will find this algorithm useful

## Data Description:
- The Data source is from Sport-Reference.com (Basketball)
- Extract all the players data from the year of 2015 to 2020 
- If the player was not in the league in 2015, then I will only evaluate the data from his starting season
- Exclude all the retired players 

## Tools:
- Use request to request the Sport-Reference url
- Use BeautifulSoup and Selenium to extract the data from Sport-Reference.com 
- Study the data using pandas and numpy
- Create python functions to extract the useful stats from the raw data of BeautifulSoup or Selenium
- Use Sklearn to construct a Linear Regression model and evaluate the accuracy of the model
- Use matplotlib and seaborn to create plots to visualize the data

## MVP
- The R-squared of the algorithm we created can be 0.7 or higher 