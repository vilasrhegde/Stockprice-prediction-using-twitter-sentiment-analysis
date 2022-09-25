
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import dill

import numpy as np
import pandas as pd
import matplotlib.pyplot as mlpt

model = dill.load(open('./randomForest.obj','rb'))

st.title ("STOCKPORT")
st.subheader("Machine Learning Project")
st.markdown("Stockprice prediction using Twitter sentiment analysis")

#------------load and alterations of data-----------------
dataframe = pd.read_pickle('cleaned_data.pkl')
train_data_start = '2007-01-01'
train_data_end = '2014-12-31'
test_data_start = '2015-01-01'
test_data_end = '2016-12-31'
train = dataframe.loc[train_data_start : train_data_end]
test = dataframe.loc[test_data_start:test_data_end]


posi=0
nega=0
for i in range (0,len(dataframe)):
    get_val=dataframe.Comp[i]
    if(float(get_val)<(-0.99)):
        nega=nega+1
    if(float(get_val>(-0.99))):
        posi=posi+1
posper=(posi/(len(dataframe)))*100
negper=(nega/(len(dataframe)))*100
#print("% of positive tweets= ",posper)
#print("% of negative tweets= ",negper)
arr=np.asarray([posper,negper], dtype=int)
# mlpt.pie(arr,labels=['Positive','Negative'])
# st.write("Polarity distribution of tweets")
# st.write(round(posper),'% Tweet were Positive')
# st.write(round(negper),'% Tweet were Negative')
# st.pyplot(mlpt)

# option = st.selectbox(
#     'Which graph you want to be shown?',
#     ('Polarity','predictions'))





#-----------calculation of sentimental scores-------------
list_of_sentiments_score = []
for date, row in train.T.iteritems():
    sentiment_score = np.asarray([dataframe.loc[date, 'Comp']])
    list_of_sentiments_score.append(sentiment_score)
numpy_dataframe_train = np.asarray(list_of_sentiments_score)

list_of_sentiments_score = []
for date, row in test.T.iteritems():
    sentiment_score = np.asarray([dataframe.loc[date, 'Comp']])
    list_of_sentiments_score.append(sentiment_score)
numpy_dataframe_test = np.asarray(list_of_sentiments_score)

y_train = pd.DataFrame(train['adj_close_price'])
y_test = pd.DataFrame(test['adj_close_price'])

rf = RandomForestRegressor()
rf.fit(numpy_dataframe_train, train['adj_close_price'])
prediction=rf.predict(numpy_dataframe_test)

idx = pd.date_range(test_data_start, test_data_end)
predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['adj_close_price'])
predictions_df['adj_close_price'] = predictions_df['adj_close_price'].apply(np.int64)
predictions_df['adj_close_price'] = predictions_df['adj_close_price'] + 4500
predictions_df['actual_value'] = test['adj_close_price']
predictions_df.columns = ['predicted_price', 'actual_price']
mlpt.plot(predictions_df)
st.pyplot(mlpt)

predictions_df['predicted_price'] = predictions_df['predicted_price'].apply(np.int64)
test['adj_close_price']=test['adj_close_price'].apply(np.int64)




prediction = rf.predict(numpy_dataframe_train)
#print("ACCURACY= ",(rf.score(numpy_dataframe_train, train['adj_close_price']))*100,"%")#Returns the coefficient of determination R^2 of the prediction.
#print(prediction)
idx = pd.date_range(train_data_start, train_data_end)
predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['Predicted Prices'])
#stocks_dataf['adj_close_price'] = stocks_dataf['adj_close_price'].apply(np.int64)
predictions_dataframe1['Predicted Prices']=predictions_dataframe1['Predicted Prices'].apply(np.int64)
print(predictions_dataframe1['Predicted Prices'])
predictions_dataframe1["Actual Prices"]=train['adj_close_price']
predictions_dataframe1.columns=['Predicted Prices','Actual Prices']
mlpt.plot(predictions_dataframe1)
st.pyplot(mlpt)
