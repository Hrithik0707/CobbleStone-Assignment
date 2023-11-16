import sys
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")





MINUTES_HOUR = 60 # minutes per hour
MINUTES_DAY = 24 * MINUTES_HOUR # minutes per day
NUMBER_DAYS_WEEK = 7 # number days a week



class StreamingMovingAverage:
    '''Moving Average algorithm'''
    def __init__(self, threshold=1.0) -> None:
        # Parameters
        self.max_deviation_from_expected = threshold
        self.min_nof_records_in_model = 3
        self.max_nof_records_in_model = 3 * self.min_nof_records_in_model

    def detect(self, timestamp: int, value: float, dumping: bool=False) -> bool:
        '''Detect if is a Anomaly'''
        self._update_state(timestamp, value)
        expected_value = self._expected_value(timestamp)
        # is there enough data and is not NaN value
        response, curr_value, deviation = False, value, 0.0
        if self._enough_data() and not np.isnan(expected_value):
            # is the value out of the boundary? when it decrease
            curr_value = expected_value
            deviation = self._standard_deviation() * self.max_deviation_from_expected
            # when it is higher than expected
            if expected_value + deviation < value or\
               expected_value - deviation > value:
                response = True
        # dumping or not
        if dumping: return (response, curr_value, deviation)
        else: return response

    def _update_state(self, timestamp: int, value: float) -> None:
        '''Update the model state'''
        # check if it is the first time the model is run or if there is a big interval between the timestamps
        if not hasattr(self, 'previous_timestamp'):
            self._init_state(timestamp)
        # update the model state
        self.previous_timestamp = timestamp
        self.data_streaming.append(value)
        # is there a lot of data? remove one record
        if len(self.data_streaming) > self.max_nof_records_in_model:
            self.data_streaming.pop(0)

    def _init_state(self, timestamp: int) -> None:
        '''Reset the parameters'''
        self.previous_timestamp = timestamp
        self.data_streaming = list()

    def _enough_data(self) -> bool:
        '''Check if there is enough data'''
        return len(self.data_streaming) >= self.min_nof_records_in_model

    def _expected_value(self, timestamp: int) -> float:
        '''Return the expected value'''
        data = self.data_streaming
        data = pd.Series(data=data, dtype=float)
        many = self.min_nof_records_in_model
        return data.rolling(many, min_periods=1).mean().iloc[-1]

    def _standard_deviation(self) -> float:
        '''Return the standard deviation'''
        data = self.data_streaming
        return np.std(data, axis=0)

    def get_state(self) -> dict:
        '''Get the state'''
        self_dict = {key: value for key, value in self.__dict__.items()}
        return pickle.dumps(self_dict, 4)

    def set_state(self, state) -> None:
        '''Set the state'''
        _self = self
        ad = pickle.loads(state)
        for key, value in ad.items():
            setattr(_self, key, value)


class AnomalyDetectionSeasonalBucket(StreamingMovingAverage):

    def __init__(self, min_buckets: int, window: int=4, min_value: int=10, threshold: float=2) -> None:
        '''Min number of messages is 1 per minute'''
        super().__init__()
        # parameters
        self.min_buckets = min_buckets
        self._num_bucket = int(np.floor(MINUTES_DAY / min_buckets))
        self.window_size = window
        self.min_value = min_value
        self.max_deviation_from_expected = threshold

    def _get_cday_cbucket(self, timestamp: dt.datetime) -> tuple:
        '''Get cday, cbucket values from timestamp'''
        # compute bucket indexes
        cday, chour, cminute = timestamp.weekday(), timestamp.hour, timestamp.minute
        cbucket = int(np.floor((chour * MINUTES_HOUR + cminute) / self.min_buckets))
        return cday, cbucket
    
    def _create_copy_data(self, timestamp: dt.datetime) -> np.array:
        '''Get data based on timestamp'''
        # compute bucket indexes, get data
        cday, cbucket = self._get_cday_cbucket(timestamp = timestamp)
        data = np.copy(self._buckets[cday, cbucket])
        data[data == 0.0] = np.nan
        return data
    
    def _update_state(self, timestamp: dt.datetime, value: float) -> None:
        '''Update the model state'''
        # check if it is the first run
        if not hasattr(self, 'previous_timestamp'):
            self._init_state(timestamp)
        # compute bucket indexes
        cday, cbucket = self._get_cday_cbucket(timestamp = timestamp)
        # shift values, empty days
        pass_days = int(np.ceil((timestamp - self._buckets_timestamp[cday, cbucket]).days / NUMBER_DAYS_WEEK))
        self._buckets[cday, cbucket] = np.roll(self._buckets[cday, cbucket], pass_days)
        self._buckets[cday, cbucket, 0:pass_days] = 0
        # update the model state
        self.previous_timestamp = timestamp
        self._buckets_timestamp[cday, cbucket] = timestamp
        self._buckets[cday, cbucket, 0] = self._buckets[cday, cbucket, 0] + value

    def _init_state(self, timestamp: dt.datetime) -> None:
        '''Reset the parameters'''
        self.previous_timestamp = timestamp
        self._buckets = np.zeros((NUMBER_DAYS_WEEK, self._num_bucket, self.window_size))
        self._buckets_timestamp = np.full((NUMBER_DAYS_WEEK, self._num_bucket), timestamp, dtype=dt.datetime)

    def _expected_value(self, timestamp: dt.datetime) -> float:
        '''Return the expected value'''
        data = self._create_copy_data(timestamp = timestamp)
        return np.nanmean(data)

    def _enough_data(self) -> bool:
        '''Check if there is enough data'''
        data = self._create_copy_data(timestamp = self.previous_timestamp)
        records = self.window_size - np.sum(np.isnan(data))
        return records >= min(self.min_nof_records_in_model, self.window_size)

    def _standard_deviation(self) -> float:
        '''Return the standard deviation'''
        # compute bucket indexes, get data
        data = self._create_copy_data(timestamp = self.previous_timestamp)
        return np.nanstd(data)

class StreamingARIMA(StreamingMovingAverage):
    '''ARIMA algorithm'''

    def __init__(self, threshold=1.0, order:tuple=(2, 1, 2)) -> None:
        # Parameters
        self.order = order
        self.model = None
        self.res = None
        self.max_deviation_from_expected = threshold
        self.min_nof_records_in_model = 10
        self.max_nof_records_in_model = 3 * self.min_nof_records_in_model

    def _expected_value(self, timestamp: int) -> float:
        '''Return the expected value'''
        if self._enough_data():
            data = self.data_streaming
            self.model = ARIMA(data, order=self.order)
            self.res = self.model.fit()
            output = self.res.forecast()
            return output[0]
        return np.nan

    def _standard_deviation(self) -> float:
        '''Return the standard deviation'''
        much = self.min_nof_records_in_model
        data = self.data_streaming
        if len(data) > much:
            data = data[-much:]
        return np.std(data, axis=0)

    def summary(self) -> None:
        '''Print the ARIMA summary'''
        if pd.notnull(self.res):
            print(self.res.summary())

def plot_anomalies(df, algorithm, parameters, dumping=False, casting=None):
    '''Plot the Streaming Data (an Anomalies)'''
    Y = df.value
    X = df.timestamp
    X_pred = df.timestamp if casting is None else X.apply(casting)
    # predict anomalies
    model = algorithm(**parameters)
    preds = [model.detect(i, v, dumping=True) for i, v in zip(X_pred, Y)]
    pred, values, stds = tuple(zip(*preds))
    # plot the results
    plt.figure(figsize=(12,4))
    model_name = algorithm.__name__
    plt.title(f'Anomaly Detection - {model_name}')
    af  = pd.DataFrame(data={'x':X, 'value':Y, 'pred':pred})
    af2 = pd.DataFrame(data={'x':X, 'value':values, 'pred':pred, 'std': stds})
    af2['ymin'] = af2['value'] - af2['std']
    af2['ymax'] = af2['value'] + af2['std']
    size = (af.pred.astype(int)+1) * 40
    sns.scatterplot(data=af, x='x', y='value', hue='pred', s=size)
    if dumping: plt.fill_between(af2.x, af2.ymin, af2.ymax, facecolor='green', alpha=0.2)
    plt.savefig("output.jpg")
    
  
dataset = sys.argv[1]
data  = pd.read_csv(dataset)
data = data.sort_values(by='timestamp', ascending=True)
mini, maxi = data['value'].min(), data['value'].max()
mm = round(data['value'].mean(), 2)
print('Mean neutral value = ', mm)
print('Minimum neutral value = ', mini)
print('Maximum neutral value = ', maxi)


if sys.argv[2]=='Moving_Average':
    parameters = {'threshold': 1.5}
    plot_anomalies(data, StreamingMovingAverage, parameters, dumping=True)
elif sys.argv[2]=='ARIMA':
    parameters = {'threshold': 1.5, 'order': (2, 1, 2)}
    plot_anomalies(data, StreamingARIMA, parameters, dumping=True)

elif sys.argv[2]=='Seasonal_bucket':
    parameters = {'min_buckets': 30, 'window': 2}
    plot_anomalies(data, AnomalyDetectionSeasonalBucket, parameters, dumping=True, casting=dt.datetime.fromtimestamp)
elif sys.argv[2]=='Isolation':
    model = IsolationForest(random_state = 0, contamination = float(0.05))
    model.fit(data[['value']])
    data['scores'] = model.decision_function(data[['value']])
    data['anomaly_score'] = model.predict(data[['value']])
    outliers = data.loc[data['anomaly_score']==-1]
    outlier_index = list(outliers.index)
    plt.figure(figsize = (16, 8))

    plt.plot(data['value'], marker = '.')
    plt.plot(outliers['value'], 'o', color = 'red', label = 'outlier')
    plt.title('Detection By Isolation Forest')

    #plt.grid()
    plt.xlabel('TimeStamp')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig("output.jpg")
else:
    model = IsolationForest(random_state = 0, contamination = float(0.05))
    model.fit(data[['value']])
    data['scores'] = model.decision_function(data[['value']])
    data['anomaly_score'] = model.predict(data[['value']])
    outliers = data.loc[data['anomaly_score']==-1]
    outlier_index = list(outliers.index)
    plt.figure(figsize = (16, 8))

    plt.plot(data['value'], marker = '.')
    plt.plot(outliers['value'], 'o', color = 'red', label = 'outlier')
    plt.title('Detection By Isolation Forest')

    #plt.grid()
    plt.xlabel('TimeStamp')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig("output.jpg")

    	
    	
    

