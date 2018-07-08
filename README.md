# 2018 KDD CUP of Fresh Air
* Task Introduction: https://biendata.com/competition/kdd_2018/
* Final Rank: https://biendata.com/competition/kdd_2018/ranking_list/


# Who We Are

* Team name —— Zugzug
* Team members (ordered by name initials):
  * Can Wang
  * Hao Li
  * Jiaquan Fang
  * Pu Zhao
  * Xiang Sun
  * Yuchen Zhang

* Organization:
  * CreditEase Big Data Innovation Center


# Code and Documentation

The code is split into four independent parts for models (each of which is bound to a learning model, operated and maintained by one of our team members), plus one part of the external data source and one integral part that merges the results all above.


Different parts of the code are structured in the *src* directory. Note that these files were originally separate projects maintained by different members, and were simply added into this integral directory after the KDD competition ended. Different parts may have different code styles and package dependencies, and some intermediate data are omitted. So there is no guarantee that these Python scripts and Jupyter notebooks will work well if you try to run them.

Each section has a README file in its own directory. The overall model description is shown in the following section.

# Model Description

For the convenience of description, some terms are defined as follows:
1. Air quality features: the values of the PM2.5, PM10, ozone, sulfur dioxide, carbon monoxide, and nitric dioxide concentration
2. Meteorology features: the values of weather, temperature, humidity, wind speed, and wind direction

## Xgboost Model #1

Code related to this model is stored in the path *src/xiangsun*. Please see for details.

At a certain time point of submission, the model evenly divides the upcoming 48 hours into 8 windows, with each time window consisting of 6 hours. Identical PM2.5, PM10, and O<sub>3</sub> concentration values are evaluated to the time nodes in one window. For a certain station, a series of Xgboost regressors do the prediction with the following features in the last 36 hours prior to the submission time point:
1. Air quality features of the station being observed;
2. Air quality and meteorology features of the *k* nearest stations;
3. Air quality and meteorology features of the nearest and the farthest stations in the diagonal directions.
Separate models are used for different prediction targets and different cities.

## Xgboost Model #2
The code of this model is stored in the path *src/lihao*. Please see for details.
In this model, a strategy of multiple model groups is utilized. The following model groups are trained:
1. Using the data of last 24 hours prior to a submission time point, to predict the air quality values after a gap of N hours, where N can be 12/24/36/48;
2. Using the data of last 24 hours prior to a submission time point, to predict the air quality values of the *i*th hour in the future, where *i* = 1, 2, 3, 4, 5, 6;
3. Using the data of last 24 hours prior to a submission time point, to predict the air quality average values of the upcoming 36 hours.

How to use the results of the models above:
1. For a certain station *j* at a certain time point within the submission range, models in group#1 provide four results (the value predicted from 12/24/36/48 hours ago). Merge these results with a group of preset weight factors, and we get *V<sub>j,1</sub>*;
2. Likewise, models in group#2 provide six results. Let *V<sub>j,2</sub>* be the average of these results. Subtract the difference between the average *V<sub>_,2</sub>* for all *j* and the average of *V<sub>_,1</sub>* for all *j* from *V<sub>j,1</sub>*;
3. Likewise, adjust the values of the first 36 time points within the submission range, using the result of the group#3, and get the final result.


Feature engineering:
1. The air quality and meteorology features of the past N hours;
2. Some statistics of these raw features, including sums, average values, standard deviations, variations, maximums, minimums, slopes, etc.
3. Only data of 5 stations are used for a certain station (itself and 4 neighbors).


## LSTM Model #3

This model is stored in the path *src/zhaopu*.

An LSTM model is trained to predict the air quality values of upcoming 48 hours of a station using the last 48 hours prior to the submission time point.

Simple air quality features and meteorology features are used. At time point *t*, the features include values of *t*-48, *t*-47,..., *t*-1, and the target series consist of values of *t*+1,*t*+2,..., *t*+48. Missing values are interpolated.


## LSTM Model #4
This model is stored in the path *src/jiaquanfang*.

This LSTM model uses the data from the meteorology station which is the nearest to the target station being observed as the features for the latter, with missing values interpolated. The self-attention mechanism is utilized in this model.

## External Data Source
The external data from AQICN (http://aqicn.org/city/beijing/) are used in our submissions. We have developed a python script (see *src/result_aqicn.py*) to download the air quality index (AQI) values of PM2.5 and PM10 from the website, interpolate the intervals (the raw data are recorded every three hours), and save the result for the merging purpose.


## Merging Strategy
 Allocate a weight factor for each of the five independent prediction results (four models and AQICN), and train a linear regressor for the group of factors to minimize the overall MSE loss between the merged and the actual results in the past 5/7/10 days.

