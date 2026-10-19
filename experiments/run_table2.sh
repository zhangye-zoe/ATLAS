#!/bin/bash
python experiments/run_table2.py --dataset pbmc --pair-ratios 0 40 100 --device cuda:0
python experiments/run_table2.py --dataset bmmc --pair-ratios 0 40 100 --device cuda:0
