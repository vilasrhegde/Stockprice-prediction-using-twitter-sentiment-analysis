# STOCKPORT
> Stockprice prediction using twitter sentiment analysis 

## Process of Prediction and Analysis:

1. Importing all the dependancies
2. Loading the dataset ```Twitter_Dataset.pkl```
3. We consider data column as our index
4. Remove dot(.) and space( ) from the dataset, which is redundant
5. We use NLTK toolkit to understand polarity of tweets
by ```nltk.download('vader_lexicon')```
6. By  feeding Tweets column it gives us the following sentiments,
    - Positive
    - Negative
    - Neutral
    - Compound
7. Plotting the PIE chart to understand percentage of positive and negative tweets that are present in our dataset.
8. The result was quite astonishing! since ratio was pretty close to each other as 55% and 44% respectively.
9. Splitting of data into train and test on the basis of years
  - Train is splitted within Jan 2007 to  Dec 2014
  - Test is splitted within Jan 2015 to Dec 2016
10. We create a list for both train and test data by calculating sentiment score of compound tweets
11. Closing price will be our label means result to compare with
12. We use Random Forest regressor as our model for prediction as ```rf = RandomForestRegressor()```
13. We plot the graph comprises of both prediction and actual data
14. And finally calculated accuracy score which we got 91.6%


