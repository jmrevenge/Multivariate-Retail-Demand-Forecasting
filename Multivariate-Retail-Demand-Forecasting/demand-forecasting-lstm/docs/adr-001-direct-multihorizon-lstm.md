# ADR 001: Direct multi-horizon forecasting with an LSTM

## Status
Accepted

## Context
The task is to forecast the next `H` days of demand for each store-product series from a
window of recent history and known drivers (price, promotions, weather, calendar). Two
structural choices had to be made: what model class, and how to produce a multi-step
forecast.

## Decision
**Model class: LSTM.** The demand process has effects that are nonlinear and interact
across time: constant-elasticity price response, a threshold/quadratic temperature
response, and promotion uplift that depends on the day of week. A recurrent model that
consumes the full feature history can represent these; a linear model on the same
features cannot. We keep a linear regression and a seasonal-naive predictor as baselines
precisely to measure whether the LSTM earns its additional complexity (see the README
eval).

**Forecast strategy: direct multi-horizon.** The network emits all `H` future days in a
single forward pass, rather than predicting one day and feeding it back autoregressively.
Recursive forecasting compounds errors: a mistake on day 1 corrupts the input for day 2.
Direct multi-horizon avoids that at the cost of not modelling the correlation between
horizon steps explicitly, which is an acceptable trade for a 7-day horizon.

## Consequences
The model is simple to train and serve (one forward pass per forecast) and is robust to
error compounding. If the horizon grew large, or if inter-step correlation mattered more,
a seq2seq decoder or a probabilistic head would be the next step.
