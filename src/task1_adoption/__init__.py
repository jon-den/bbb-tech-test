"""Task 1: Camzyos adoption modelling.

Discrete-time hazard model of Camzyos initiation among Disopyramide-
experienced oHCM patients. Pipeline in scripts/task1_adoption/.

Modules
-------
config        : Clinical codes, feature sets, dataclass configs.
data_loading  : Load raw synthetic_data/ CSVs into a typed RawData namespace.
dataset       : Person-month dataset with risk-set conditioning, censoring, features.
models        : DiscreteHazardGLM (cloglog/logit), GBMHazardBenchmark, MarginalRateModel.
evaluation    : Brier decomposition, time-dependent AUC, count-level calibration.
selection     : StabilitySelector (bootstrap + L1), LASSO path plot.
"""
