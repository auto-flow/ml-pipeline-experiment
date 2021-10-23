#!/usr/bin/env bash
#python experiments/run_hyperopt.py   146594 pipeline 100 250 20
#python experiments/run_hyperopt.py   189864 pipeline 100 250 20
#python experiments/run_ultraopt.py   146594 pipeline 100 250 20
#python experiments/run_ultraopt.py   189864 pipeline 100 250 20
python experiments/run_smac.py   189864 pipeline 100 250 20
python experiments/run_smac.py   146594 pipeline 100 250 20


