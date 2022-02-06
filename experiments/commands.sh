#!/usr/bin/env bash
#python experiments/run_hyperopt.py   146594 pipeline 100 250 20
#python experiments/run_hyperopt.py   189864 pipeline 100 250 20
#python experiments/run_ultraopt.py   146594 pipeline 100 250 20
#python experiments/run_ultraopt.py   189864 pipeline 100 250 20
#python experiments/run_smac.py   189864 pipeline 100 250 20
#python experiments/run_smac.py   146594 pipeline 100 250 20
#python experiments/run_ultraopt.py   146594 bagging 10 500 50
#python experiments/run_hyperopt.py   146594 bagging 10 500 50
#python experiments/run_smac.py   146594 bagging 10 500 50
#python experiments/run_hyperopt.py   146594 pipeline 100 250 20
#python experiments/run_hyperopt.py   189864 pipeline 100 250 20
#python experiments/run_ultraopt.py   189864 bagging 100 500 50
#python experiments/run_hyperopt.py   189864 bagging 100 500 50
#python experiments/run_ultraopt.py   189863 pipeline 100 500 50
#python experiments/run_hyperopt.py   189863 pipeline 100 500 50
#python experiments/run_ultraopt.py   146594 pipeline 100 500 50
#python experiments/run_hyperopt.py   146594 pipeline 100 500 50
#python experiments/run_ultraopt.py   189864 pipeline 100 500 50
#python experiments/run_hyperopt.py   189864 pipeline 100 500 50
export LOCAL_SEARCH=false
export PYTHONPATH=/data/Project/AutoML/ML-Pipeline-Experiment:/data/Project/AutoML/ultraopt:/data/Project/AutoML/SMAC3
#python experiments/run_hyperopt.py   146594 pipeline 100 500 50
#python experiments/run_smac.py   146594 pipeline 100 500 50
#python experiments/run_smac.py   189863 pipeline 100 500 50
#python experiments/run_smac.py   189864 pipeline 100 500 50
export repetitions=30
#export repetitions=50
export multivariate=false
#python experiments/run_ultraopt.py   189864 bagging $repetitions 500 50 $multivariate
#python experiments/run_ultraopt.py   146594 bagging $repetitions 500 50 $multivariate
#python experiments/run_ultraopt.py   189863 bagging $repetitions 500 50 $multivariate
python experiments/run_ultraopt.py   189864 pipeline $repetitions 500 50 $multivariate
python experiments/run_ultraopt.py   146594 pipeline $repetitions 500 50 $multivariate
python experiments/run_ultraopt.py   189863 pipeline $repetitions 500 50 $multivariate
#
#export multivariate=true
#
#python experiments/run_ultraopt.py   189864 bagging $repetitions 500 50 $multivariate
#python experiments/run_ultraopt.py   146594 bagging $repetitions 500 50 $multivariate
#python experiments/run_ultraopt.py   189863 bagging $repetitions 500 50 $multivariate
#python experiments/run_ultraopt.py   189864 pipeline $repetitions 500 50 $multivariate
#python experiments/run_ultraopt.py   146594 pipeline $repetitions 500 50 $multivariate
#python experiments/run_ultraopt.py   189863 pipeline $repetitions 500 50 $multivariate

#python experiments/run_optuna.py   189864 bagging $repetitions 500 50
#python experiments/run_optuna.py   146594 bagging $repetitions 500 50
#python experiments/run_optuna.py   189863 bagging $repetitions 500 50
#python experiments/run_optuna.py   189864 pipeline $repetitions 500 50
#python experiments/run_optuna.py   146594 pipeline $repetitions 500 50
#python experiments/run_optuna.py   189863 pipeline $repetitions 500 50


python experiments/run_bohb_kde.py   189864 bagging $repetitions 500 50
python experiments/run_bohb_kde.py   146594 bagging $repetitions 500 50
python experiments/run_bohb_kde.py   189863 bagging $repetitions 500 50
python experiments/run_bohb_kde.py   189864 pipeline $repetitions 500 50
python experiments/run_bohb_kde.py   146594 pipeline $repetitions 500 50
python experiments/run_bohb_kde.py   189863 pipeline $repetitions 500 50

python experiments/run_random.py   189864 bagging $repetitions 500 50
python experiments/run_random.py   146594 bagging $repetitions 500 50
python experiments/run_random.py   189863 bagging $repetitions 500 50
python experiments/run_random.py   189864 pipeline $repetitions 500 50
python experiments/run_random.py   146594 pipeline $repetitions 500 50
python experiments/run_random.py   189863 pipeline $repetitions 500 50

exit 0

export max_groups=3
python experiments/run_ultraopt.py   189864 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   146594 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189863 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189864 bagging $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   146594 bagging $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189863 bagging $repetitions 500 50 $max_groups

export max_groups=4
python experiments/run_ultraopt.py   189864 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   146594 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189863 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189864 bagging $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   146594 bagging $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189863 bagging $repetitions 500 50 $max_groups

export max_groups=5
python experiments/run_ultraopt.py   189864 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   146594 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189863 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189864 bagging $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   146594 bagging $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189863 bagging $repetitions 500 50 $max_groups

export max_groups=6
python experiments/run_ultraopt.py   189864 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   146594 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189863 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189864 bagging $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   146594 bagging $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189863 bagging $repetitions 500 50 $max_groups

export max_groups=8
python experiments/run_ultraopt.py   189864 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   146594 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189863 pipeline $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189864 bagging $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   146594 bagging $repetitions 500 50 $max_groups
python experiments/run_ultraopt.py   189863 bagging $repetitions 500 50 $max_groups

#python experiments/run_ultraopt.py   146594 bagging 100 500 50
#python experiments/run_ultraopt.py   189863 bagging 100 500 50
#python experiments/run_ultraopt.py   189864 bagging 100 500 50
#python experiments/run_hyperopt.py   146594 bagging 100 500 50
#python experiments/run_hyperopt.py   189863 bagging 100 500 50
#python experiments/run_hyperopt.py   189864 bagging 100 500 50
#python experiments/run_smac.py   189863 bagging  30 500 50
#python experiments/run_smac.py   189864 bagging  30 500 50
#python experiments/run_smac.py   146594 bagging  30 500 50
#echo "finish bagging"
#python experiments/run_smac.py   189863 pipeline 30 200 50
#python experiments/run_smac.py   189864 pipeline 30 200 50
#python experiments/run_smac.py   146594 pipeline 30 200 50






