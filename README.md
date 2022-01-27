# NAMLProject - Price Prediction

In [data](data) can be found some CSV files of BTC/EUR trade values taken from Binance API with different timeframes.  
The format is Timestamp - OHLC+V ([Open-High-Low-Close + Volume](https://en.wikipedia.org/wiki/Open-high-low-close_chart)).

### Timestamp to Datetime conversion

To convert from timestamp to datetime, use this line of code

```python
data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms')
```
