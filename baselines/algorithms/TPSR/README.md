# uDSR

## running steps:

1. run gen_csv_datasets.py in main project folder
2. run train/train.py for training models
3. run TPSR/run.py --eval_mcts_on_pmlb True \
                   --lam 0.1 \
                   --horizon 200 \
                   --width 3 \
                   --num_beams 1 \
                   --rollout 3 \
                   --no_seq_cache False \
                   --no_prefix_cache True \
                   --max_input_points 200 \
                   --max_number_bags 10 \
                   --save_results True