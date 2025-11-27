import time
import random
import warnings
import numpy as np
import pandas as pd
from loguru import logger
from lightgbm import LGBMRegressor
from econml.dml import CausalForestDML
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)
start_time = time.time()

logger.debug('...Processing dataset...')
store_id = 1
open_hour, close_hour = 6, 23
start_day = '2024-03-01 00:00:00'
end_day = '2024-12-03 23:59:59'
data = pd.read_csv(f"dataset_store{store_id}_.csv", index_col=0)
data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S+00:00')
data.index = data.index.tz_localize('UTC').tz_convert('Europe/Lisbon')
data = data[data.index >= start_day]
data = data[data.index <= end_day]
n_hours = data.index.hour.unique().shape[0]

logger.debug('...Targets for treatment and outcome model...')
target_y, target_t = 'consumption_total', 'discount'
data[target_t] = (data[target_t] * 100).astype(int)

logger.debug('...Calculate lags...')
data[f'{target_y}_lag1d'] = data[target_y].shift(n_hours)
data[f'{target_y}_lag7d'] = data[target_y].shift(n_hours * 7)
data[f'{target_t}_lag1h'] = data[target_t].shift(1)
data[f'{target_t}_lag2h'] = data[target_t].shift(2)
data[f'{target_t}_lag1d'] = data[target_t].shift(n_hours)
data = data[n_hours * max(1, 7):]

logger.debug('...Creating additional features...')
data['date'] = data.index.date
data['discounts_so_far_today'] = (
    data.groupby('date')['discount']
    .apply(lambda s: (s.shift() > 0).cumsum())
    .reset_index(level=0, drop=True)
)
data['is_max_price_hour'] = (data['market_price'] == data.groupby('date')['market_price'].transform('max')).astype(int)
data['is_min_price_hour'] = (data['market_price'] == data.groupby('date')['market_price'].transform('min')).astype(int)
price_bins, price_labels = [-np.inf, 10, 50, 100, 150, 200, np.inf], [0, 1, 2, 3, 4, 5]
data = data.dropna(subset=['market_price'])
data['price_bucket'] = pd.cut(data['market_price'], bins=price_bins, labels=price_labels)
data['price_bucket'] = data['price_bucket'].astype(int)
data.drop(columns=['date'], inplace=True)

logger.debug('...Calculate neighboring hour lags for outcome variable...')
y_lag_hours = 2
for i in range(1, y_lag_hours + 1):
    data[f'{target_y}_-{i}h'] = data[target_y].shift(i)
    data[f'{target_y}_+{i}h'] = data[target_y].shift(-i)

logger.debug('...Remove unavailable service days, holidays, non-business hours...')
data['hour'] = data.index.hour
data = data[(data.hour > open_hour) & (data.hour < close_hour)]
data = data[(data['availability_service'] == 1) & (data['availability_discounts'] == 1) & (data['holiday'] == 0)]
list_hours = data.index.hour.unique().sort_values().to_list()
n_hours = len(list_hours)

logger.debug('...Defining model covariates...')
common_causes = [
    'temperature_hist',
    f'{target_y}_lag1d',
    f'{target_y}_lag7d',
    'forecast_total',
    f'{target_t}_lag1h',
    f'{target_t}_lag2h',
    f'{target_t}_lag1d',
    'discounts_so_far_today',
    'is_max_price_hour',
    'is_min_price_hour',
    'price_bucket'
]
modifiers = ['hour', 'weekday']

logger.debug('...Remove missing data...')
logger.debug(f'Number of observations: {data.shape[0]}')
y_lags_neg = [f'{target_y}_{i}h' for i in range(-y_lag_hours, 0)]
y_lags_pos = [f'{target_y}_+{i}h' for i in range(1, y_lag_hours + 1)]
y_lags = y_lags_neg + y_lags_pos
data = data[[target_y, target_t, f'{target_t}_category'] + y_lags + common_causes + modifiers]
data.dropna(inplace=True)
logger.debug(f'Number of observations (w/o missing data): {data.shape[0]}')

# logger.debug('...Scaling features...')
# data_ = data[common_causes + modifiers]
# data_columns = data_.columns
# data_ = StandardScaler().fit_transform(data_)
# data_df = pd.DataFrame(data_, columns=data_columns, index=data.index)
# data_df[target_y] = data[target_y]
# data_df[target_t] = data[target_t]
# for lag in y_lags:
#     data_df[lag] = data[lag]
# data = data_df.copy()
# n_days = int(data.index.shape[0] / n_hours)

logger.debug('...Get outcome Y and treatment T, and covariates W and X...')
Y = data[y_lags_neg + [target_y] + y_lags_pos].values
T = data[[target_t]].values
W = data[common_causes].values
X = data[modifiers].values

logger.debug('...Fitting Causal model...')
cv = KFold(n_splits=5, shuffle=True)
model_t = LGBMRegressor(n_estimators=300, max_depth=4, verbose=-1)
model_y = RandomForestRegressor(n_estimators=300, max_depth=7)
model = CausalForestDML(model_y=model_y, model_t=model_t, cv=cv, mc_iters=10, random_state=42,
                        n_estimators=300, max_depth=5)
model.fit(Y, T, W=W, X=X, cache_values=True)

logger.debug('...Model summary...')
logger.debug(model.summary())

logger.debug('...Computing marginal effects...')
effect = model.effect(X=X, T0=np.zeros(T.shape), T1=T)
effect_interval = model.effect_interval(X=X, T0=np.zeros(T.shape), T1=T)
marginal_effect = model.const_marginal_effect(X=X)
logger.debug(f'\n{marginal_effect}')

logger.debug('...Computing effect by neighboring hour...')
effect_df = pd.DataFrame(effect)
effect_df['discount'] = T
dict_names = {i + y_lag_hours: f'H{i}' for i in range(-y_lag_hours, 0)}
dict_names.update({len(dict_names): 'H'})
dict_names.update({i + y_lag_hours: f'H+{i}' for i in range(1, y_lag_hours + 1)})
effect_df = effect_df.rename(columns=dict_names)
effect_df_melted = effect_df.melt(id_vars='discount', var_name='Hour', value_name='Treatment Effect')
effect_df_melted = effect_df_melted[effect_df_melted['discount'] > 0]
logger.debug(f'\n{effect_df_melted.groupby(["Hour", "discount"]).mean()}')

logger.debug('...Computing effect time series...')
effect_multiple = pd.DataFrame()
for j, lag in enumerate([f'H{i}' for i in range(-y_lag_hours, 0)]):
    effect_multiple[lag] = effect_df[lag].shift(-int(lag.split('-')[1]))
for j, lag in enumerate([f'H+{i}' for i in range(1, y_lag_hours + 1)]):
    effect_multiple[lag] = effect_df[lag].shift(int(lag.split('+')[1]))
cross_contrib = effect_multiple.sum(axis=1)
all_contrib = cross_contrib + effect_df['H']
results = pd.DataFrame(index=data.index, data={'effect': all_contrib.values, 'demand': data[target_y]})
logger.debug(f'\n{results}')

logger.debug(f"Execution time: {(time.time() - start_time)/60:.2f} minutes")
