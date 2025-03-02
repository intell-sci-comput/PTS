# TPSR

Official Implementation of the **state-of-the-art** baseline **[Transformer-based Planning for Symbolic Regression](https://arxiv.org/abs/2303.06833)** (NeurIPS 2023) by Parshin Shojaee, Kazem Meidani, Amir Barati Farimani, and Chandan K. Reddy. TPSR introduces a novel approach that incorporates Monte Carlo Tree Search into transformer decoding for symbolic regression, significantly improving accuracy-complexity trade-offs.

## Running Steps:

1. Run `gen_csv_datasets.py` in main project folder
2. Run `train/train.py` for training models
3. Run 
```bash
TPSR/run.py --eval_mcts_on_pmlb True \
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
```

## Citation

```bibtex
@inproceedings{shojaee2023transformer,
  title={Transformer-based Planning for Symbolic Regression},
  author={Shojaee, Parshin and Meidani, Kazem and Farimani, Amir Barati and Reddy, Chandan K.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```