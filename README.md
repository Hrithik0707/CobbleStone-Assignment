# CobbleStone-Assignment

## Project Title:
Efficient Data Stream Anomaly Detection

## Project Description:
Your task is to develop a Python script capable of detecting anomalies in a continuous data stream. This stream, simulating real-time sequences of floating-point numbers, could represent various metrics such as financial transactions or system metrics. Your focus will be on identifying unusual patterns, such as exceptionally high values or deviations from the norm.

## Objectives:

* Algorithm Selection: Identify and implement a suitable algorithm for anomaly detection, capable of adapting to concept drift and seasonal variations.
* Data Stream Simulation: Design a function to emulate a data stream, incorporating regular patterns, seasonal elements, and random noise.
* Anomaly Detection: Develop a real-time mechanism to accurately flag anomalies as the data is streamed.
Optimization: Ensure the algorithm is optimized for both speed and efficiency.
* Visualization: Create a straightforward real-time visualization tool to display both the data stream and any detected anomalies.


## How to Run the project.
* git clone https://github.com/Hrithik0707/CobbleStone-Assignment.git or download zip.
* create a virtual environment using "virtualenv myenv"
* pip install -r requirements.txt.
* python3 main.py {yourfile.csv} {algorithm}
  eg. python3 main.py cpu4.csv Isolation

## Solution. 
  ### Assumptions. 
    * Input should be in .csv format with necessary two columns as the "timestamp"(index) and "value"(floating point).
    * Four Algorithms , Moving Average  (Moving_Average) , ARIMA (ARIMA) , Seasonal Moving Average (Seasonal_bucket) , Isolation Forest (Isolation) 
    * Output visulization is showin in output.jpg.

## Algorithm and Time Complexity


### Isolation Forest

Isolation Forest is an ensemble learning algorithm primarily used for anomaly detection. It isolates anomalies instead of profiling normal data points. It uses a forest of random trees, where each tree isolates observations by randomly selecting a feature and then randomly selecting a split value for that feature.

* Effective in high-dimensional datasets.
* Suitable for unsupervised anomaly detection where anomalies are few and different.
* Efficient with large datasets.
* Capable of handling complex and high-dimensional data.
* Doesn't require defining a 'normal' profile, making it versatile for different data types.

Cons:

* Random nature might lead to variability in detection.
* Less effective if anomalies form a dense cluster.


Time Complexity:

The average time complexity of building an individual tree in an Isolation Forest is O(n log n), where n is the number of samples. However, since the algorithm typically works on a subset of the data (a sample), this can be reduced. The overall complexity depends on the number of trees in the forest and the subsample size.
Prediction time complexity is O(h * t), where h is the height of the trees (logarithmic with respect to the sample size) and t is the number of trees.

### Moving Average
  A moving average smooths time series data to identify the underlying trend. It calculates the average of different subsets of the full dataset.

* Common in financial data analysis, stock market trends, and economic indicators.
* Useful for smoothing short-term fluctuations and highlighting longer-term trends or cycles.
* Simple and intuitive, easy to understand and implement.
* Effective in reducing 'noise' in data.

Cons:

* Lagging indicator: it reacts to events only after they have occurred.
* Not suitable for predicting future values beyond short-term trends.


Time Complexity:

The time complexity of a simple moving average is O(n), where n is the length of the time series. Each new output value is computed by averaging a fixed number of past values, which is a linear operation.

### ARIMA
ARIMA models the next step in a sequence as a linear function of the observations and differencing of observations at previous time steps. It combines autoregressive (AR), differencing (I), and moving average (MA) models.

* Widely used for forecasting in finance, economics, and weather.
* Suitable for time series data with a clear trend or seasonal patterns.


* Highly customizable: can model various types of time series data.
* Good for short-term forecasting.
Cons:

* Requires the data to be stationary (constant mean and variance over time).
* Can be complex to configure and understand.
* Not suitable for high-dimensional datasets or where the data doesn't follow time series assumptions.

Time Complexity:

The time complexity of fitting an ARIMA model is more complex and can range from O(n) to O(n²), depending on the implementation, the order of the model (p, d, q), and the optimizations used. The (p, d, q) parameters represent the autoregressive, differencing, and moving average components, respectively.
For large values of p and q, the complexity increases as the algorithm needs to estimate a higher number of parameters.

S
### Seasonal Moving Average
easonal Moving Average is a variation of the moving average technique tailored for time series data with seasonal patterns. It involves calculating moving averages for each season separately. This method is particularly useful in scenarios where the data exhibits regular and predictable changes that repeat over each season.

Time Complexity:

The time complexity of a seasonal moving average is still O(n), where n is the length of the time series. This is because the computation for each point in the series is a simple average of a fixed number of points (the number of seasons), akin to the standard moving average.
The operation remains linear since the process of averaging a set number of points does not change in complexity as the dataset size increases.

### Summary

* Isolation Forest offers a good balance between performance and scalability, especially for high-dimensional datasets, but its complexity can increase with the number of trees.
* Moving Average is highly scalable due to its linear complexity, making it efficient for large datasets, but it's limited in its application to basic trend analysis.
* ARIMA can be computationally intensive, especially for large datasets or higher order models. It's less scalable compared to the other two methods but offers more comprehensive forecasting capabilities.
* The Seasonal Moving Average is efficient and straightforward for time series data with clear seasonal patterns. Its linear time complexity (O(n)) makes it scalable and suitable for large datasets.
* For this task Isolation Forest is been choosen as default.
