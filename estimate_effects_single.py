import time
import random
import warnings
import numpy as np
import pandas as pd
from loguru import logger
import statsmodels.api as sm
from dowhy import CausalModel
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

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
data[f'{target_y}_lag1h'] = data[target_y].shift(1)
data[f'{target_t}_lag1h'] = data[target_t].shift(1)
data[f'{target_t}_lag2h'] = data[target_t].shift(2)
data[f'{target_t}_lag1d'] = data[target_t].shift(n_hours)
data = data[n_hours * 7:]

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

logger.debug('...Remove unavailable service days, holidays, non-business hours...')
data['hour'] = data.index.hour
data = data[(data.hour > open_hour) & (data.hour < close_hour)]
data = data[(data['availability_service'] == 1) & (data['availability_discounts'] == 1) & (data['holiday'] == 0)]
list_hours = data.index.hour.unique().sort_values().to_list()
n_hours = len(list_hours)

logger.debug('...Defining model covariates...')
common_causes = [
    'temperature_hist',
    f'{target_y}_lag1h', f'{target_y}_lag1d', f'{target_y}_lag7d',
    f'{target_t}_lag1h', f'{target_t}_lag2h', f'{target_t}_lag1d',
    'is_max_price_hour', 'is_min_price_hour', 'price_bucket',
    'discounts_so_far_today',
    'forecast_total',
]
modifiers = ['hour', 'weekday']

logger.debug('...Remove missing data...')
logger.debug(f'Number of observations: {data.shape[0]}')
data = data[[target_y, target_t, f'{target_t}_category'] + common_causes + modifiers]
data.dropna(inplace=True)
logger.debug(f'Number of observations (w/o missing data): {data.shape[0]}')

logger.debug('...Simple regression approach...')
simple_reg = True
if simple_reg:
    logger.debug('...Compute interactions...')
    data_x = data[[target_t] + modifiers + common_causes].copy()
    data_x = data_x.T.drop_duplicates().T
    interactions = ['hour', 'weekday']
    for inter in interactions:
        if inter in data_x.columns:
            data_x[f'{target_t}_x_{inter}'] = data_x[target_t] * data_x[inter]

    logger.debug('...Compute OLS...')
    data_x = sm.add_constant(data_x)
    model_reg = sm.OLS(data[target_y], data_x).fit()
    logger.debug(f'\n{model_reg.summary()}')

logger.debug('...Instantiate DoWhy model...')
model = CausalModel(
    data=data,
    treatment=target_t,
    outcome=target_y,
    common_causes=common_causes,
    effect_modifiers=modifiers,
)
logger.debug(f'Modifiers: {modifiers}')
logger.debug(f'Common causes: {common_causes}')

logger.debug('...Identifying estimand...')
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
logger.debug(identified_estimand)

logger.debug('...Fitting model...')
cv = KFold(n_splits=3, shuffle=True)
method_name = "backdoor.econml.dml.CausalForestDML"
method_params = {
                'init_params': {'model_y': LGBMRegressor(max_depth=7, n_estimators=300, random_state=42, verbose=-1),
                                'model_t': LGBMRegressor(max_depth=4, n_estimators=300, random_state=42, verbose=-1),
                                'cv': cv,
                                'random_state': 42,
                                'mc_iters': 5,
                                'n_estimators': 300, 'max_depth': 5,
                                },
                'fit_params': {'cache_values': True},
                }
estimate = model.estimate_effect(identified_estimand,
                                 method_name=method_name,
                                 method_params=method_params,
                                 control_value=0,
                                 treatment_value=[5, 25, 50],
                                 confidence_intervals=True,
                                 )
logger.debug(estimate)

logger.debug('...Model summary...')
logger.debug(f'\n{estimate._estimator_object.summary()}')
logger.debug(f'Causal effect: {estimate.value}')

logger.debug('...Computing effect time series...')
X = data[modifiers].values if modifiers else None
effect = estimate._estimator_object.effect(X=X, T0=np.zeros(data[target_t].shape), T1=data[target_t])
effect_interval = estimate._estimator_object.effect_interval(X=X, T0=np.zeros(data[target_t].shape), T1=data[target_t])
results = pd.DataFrame(index=data.index, data={'effect': effect.ravel(), 'effect_lb': effect_interval[0].ravel(),
                                               'effect_ub': effect_interval[1].ravel(), 'demand': data[target_y]})
logger.debug(f'\n{results}')

logger.debug('...Computing marginal effects...')
comb = np.array(np.meshgrid(data['hour'].unique(), data['weekday'].unique())).T.reshape(-1, 2)
comb_ = np.array(np.meshgrid(data['hour'].unique(), data.index.weekday.unique())).T.reshape(-1, 2)
marginal_effect = estimate._estimator_object.const_marginal_effect(comb)
hours__ = comb[:, 0]
weekdays___ = comb_[:, 1]
marginal_results = pd.DataFrame({'Hour': hours__,
                                 'Weekday': weekdays___,
                                 'Effect': marginal_effect.ravel()})
marginal_results = marginal_results.pivot(index='Weekday', columns='Hour', values='Effect')
marginal_results.columns = data.index.hour.unique().sort_values().to_list()
marginal_results.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
logger.debug(f'\n{marginal_results}')

logger.debug('...Refutation tests...')
refutation_methods = ["random_common_cause", "placebo_treatment_refuter", "data_subset_refuter"]
for method in refutation_methods:
    result = model.refute_estimate(identified_estimand, estimate, method_name=method, show_progress_bar=True)
    logger.debug(result)

end_time = time.time()
logger.debug(f'Execution time: {end_time - start_time} seconds')
