# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import argparse
from symbolicregression.envs import ENVS
from symbolicregression.utils import bool_flag


def get_parser():
    """
    Generate a parameters parser.
    """
   
    # parse parameters
    parser = argparse.ArgumentParser(description="Function prediction", add_help=False)
    
    parser.add_argument("--backbone_model", type=str, default="e2e", help="e2e or nesymres")

     ## args MCTS
    parser.add_argument(
        "--seed", type=int, default=23, help="seed for the experiments MCTS"
    )
    parser.add_argument(
        "--width", type=int, default=3, help="k max "
    )
    parser.add_argument("--horizon", default=200, type=int)
    parser.add_argument("--rollout", default=3, type=int)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument('--train_value', type=bool, default=False, help="Train the value function.")
    parser.add_argument('--no_seq_cache', type=bool, default=True)
    parser.add_argument('--no_prefix_cache' ,type=bool, default=True)
    parser.add_argument('--sample_only', type=bool, default=False)
    parser.add_argument("--ucb_constant",type=float,default=1.,help="Beta Constant in UCB",)
    parser.add_argument("--ucb_base",type=float,default=10.,help="Cbase in UCB",)
    parser.add_argument("--uct_alg", default="uct", choices=["uct", "p_uct", "var_p_uct"],
                        help="The UCT algorithm to use."
                             "`uct` is the original UCT algorithm,"
                             "`p_uct` is the UCT algorithm with PUCT,"
                             "and `var_p_uct` is the UCT algorithm with variable PUCT.")
    parser.add_argument("--ts_mode", default="best", choices=["best", "sample"], help="Tree search mode within the evaluation step. `best` uses beam search, `sample` uses sampling.")
    parser.add_argument("--alg", default="mcts", choices=["mcts", "mcts-multi", "bs", "sample"])
    parser.add_argument("--entropy_weighted_strategy", default='none', choices=['none', 'linear', 'linear_with_minimum'])
    

    # main parameters
    parser.add_argument(
        "--dump_path", type=str, default="", help="Experiment dump path"
    )
    parser.add_argument(
        "--refinements_types",
        type=str,
        default="method=BFGS_batchsize=256_metric=/_mse",
        help="What refinement to use. Should separate by _ each arg and value by =. None does not do any refinement",
    )

    parser.add_argument(
        "--eval_dump_path", type=str, default=None, help="Evaluation dump path"
    )
    parser.add_argument(
        "--save_results", type=bool, default=True, help="Should we save results?"
    )

    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument(
        "--print_freq", type=int, default=100, help="Print every n steps"
    )
    parser.add_argument(
        "--save_periodic",
        type=int,
        default=25,
        help="Save the model periodically (0 to disable)",
    )
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")

    # float16 / AMP API
    parser.add_argument(
        "--fp16", type=bool_flag, default=False, help="Run model with float16"
    )
    parser.add_argument(
        "--amp",
        type=int,
        default=-1,
        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.",
    )
    parser.add_argument(
        "--rescale", type=bool, default=True, help="Whether to rescale at inference.",
    )

    # model parameters
    parser.add_argument(
        "--embedder_type",
        type=str,
        default="LinearPoint",
        help="[TNet, LinearPoint, Flat, AttentionPoint] How to pre-process sequences before passing to a transformer.",
    )

    parser.add_argument(
        "--emb_emb_dim", type=int, default=64, help="Embedder embedding layer size"
    )
    parser.add_argument(
        "--enc_emb_dim", type=int, default=512, help="Encoder embedding layer size"
    )
    parser.add_argument(
        "--dec_emb_dim", type=int, default=512, help="Decoder embedding layer size"
    )
    parser.add_argument(
        "--n_emb_layers", type=int, default=1, help="Number of layers in the embedder",
    )
    parser.add_argument(
        "--n_enc_layers",
        type=int,
        default=2,
        help="Number of Transformer layers in the encoder",
    )
    parser.add_argument(
        "--n_dec_layers",
        type=int,
        default=16,
        help="Number of Transformer layers in the decoder",
    )
    parser.add_argument(
        "--n_enc_heads",
        type=int,
        default=16,
        help="Number of Transformer encoder heads",
    )
    parser.add_argument(
        "--n_dec_heads",
        type=int,
        default=16,
        help="Number of Transformer decoder heads",
    )
    parser.add_argument(
        "--emb_expansion_factor",
        type=int,
        default=1,
        help="Expansion factor for embedder",
    )
    parser.add_argument(
        "--n_enc_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer encoder",
    )
    parser.add_argument(
        "--n_dec_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer decoder",
    )

    parser.add_argument(
        "--norm_attention",
        type=bool_flag,
        default=False,
        help="Normalize attention and train temperaturee in Transformer",
    )
    parser.add_argument("--dropout", type=float, default=0, help="Dropout")
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0,
        help="Dropout in the attention layer",
    )
    parser.add_argument(
        "--share_inout_emb",
        type=bool_flag,
        default=True,
        help="Share input and output embeddings",
    )
    parser.add_argument(
        "--enc_positional_embeddings",
        type=str,
        default=None,
        help="Use none/learnable/sinusoidal/alibi embeddings",
    )
    parser.add_argument(
        "--dec_positional_embeddings",
        type=str,
        default="learnable",
        help="Use none/learnable/sinusoidal/alibi embeddings",
    )

    parser.add_argument(
        "--env_base_seed",
        type=int,
        default=0,
        help="Base seed for environments (-1 to use timestamp seed)",
    )
    parser.add_argument(
        "--test_env_seed", type=int, default=1, help="Test seed for environments"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of sentences per batch"
    )
    parser.add_argument(
        "--batch_size_eval",
        type=int,
        default=64,
        help="Number of sentences per batch during evaluation (if None, set to 1.5*batch_size)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam_inverse_sqrt,warmup_updates=10000",
        help="Optimizer (SGD / RMSprop / Adam, etc.)",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=0.5,
        help="Clip gradients norm (0 to disable)",
    )
    parser.add_argument(
        "--n_steps_per_epoch", type=int, default=3000, help="Number of steps per epoch",
    )
    parser.add_argument(
        "--max_epoch", type=int, default=100000, help="Number of epochs"
    )
    parser.add_argument(
        "--stopping_criterion",
        type=str,
        default="",
        help="Stopping criterion, and number of non-increase before stopping the experiment",
    )

    parser.add_argument(
        "--accumulate_gradients",
        type=int,
        default=1,
        help="Accumulate model gradients over N iterations (N times larger batch sizes)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of CPU workers for DataLoader",
    )

    parser.add_argument(
        "--train_noise_gamma",
        type=float,
        default=0.0,
        help="Should we train with additional output noise",
    )

    parser.add_argument(
        "--ablation_to_keep",
        type=str,
        default=None,
        help="which ablation should we do",
    )

    parser.add_argument(
        "--max_input_points",
        type=int,
        default=200,
        help="split into chunks of size max_input_points at eval",
    )
    parser.add_argument(
        "--n_trees_to_refine", type=int, default=10, help="refine top n trees"
    )

    # export data / reload it
    parser.add_argument(
        "--export_data",
        type=bool_flag,
        default=False,
        help="Export data and disable training.",
    )
    parser.add_argument(
        "--reload_data",
        type=str,
        default="",
        # default="functions,../symbolicregression/dump/debug/data_train/data.prefix,../symbolicregression/dump/debug/data_train/data.prefix,",
        help="Load dataset from the disk (task1,train_path1,valid_path1,test_path1;task2,train_path2,valid_path2,test_path2)",
    )
    parser.add_argument(
        "--reload_size",
        type=int,
        default=-1,
        help="Reloaded training set size (-1 for everything)",
    )
    parser.add_argument(
        "--batch_load",
        type=bool_flag,
        default=False,
        help="Load training set by batches (of size reload_size).",
    )

    # environment parameters
    parser.add_argument(
        "--env_name", type=str, default="functions", help="Environment name"
    )
    ENVS[parser.parse_known_args()[0].env_name].register_args(parser)

    # 
    parser.add_argument("--tasks", type=str, default="functions", help="Tasks")

    # beam search configuration
    parser.add_argument(
        "--beam_eval",
        type=bool_flag,
        default=True,
        help="Evaluate with beam search decoding.",
    )
    parser.add_argument(
        "--max_generated_output_len",
        type=int,
        default=200,
        help="Max generated output length",
    )
    parser.add_argument(
        "--beam_eval_train",
        type=int,
        default=0,
        help="At training time, number of validation equations to test the model on using beam search (-1 for everything, 0 to disable)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size, default = 1 (greedy decoding)",
    )
    parser.add_argument(
        "--beam_type", type=str, default="sampling", help="Beam search or sampling",
    )
    parser.add_argument(
        "--beam_temperature",
        type=int,
        default=0.1,
        help="Beam temperature for sampling",
    )

    parser.add_argument(
        "--beam_length_penalty",
        type=float,
        default=1,
        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.1,
        help="Lambda i nMCTS",
    )
    parser.add_argument(
        "--beam_early_stopping",
        type=bool_flag,
        default=True,
        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.",
    )
    parser.add_argument("--beam_selection_metrics", type=int, default=1)

    parser.add_argument("--max_number_bags", type=int, default=10)

    # reload pretrained model / checkpoint
    parser.add_argument(
        "--reload_model", type=str, default="", help="Reload a pretrained model"
    )
    parser.add_argument(
        "--reload_checkpoint", type=str, default="", help="Reload a checkpoint"
    )

    # evaluation
    parser.add_argument(
        "--validation_metrics",
        type=str,
        default="r2_zero,r2,accuracy_l1_biggio,accuracy_l1_1e-3,accuracy_l1_1e-2,accuracy_l1_1e-1,_complexity",
        help="What metrics should we report? accuracy_tolerance/_l1_error/r2/_complexity/_relative_complexity/is_symbolic_solution",
    )

    parser.add_argument(
        "--debug_train_statistics",
        type=bool,
        default=False,
        help="whether we should print infos distributions",
    )

    parser.add_argument(
        "--eval_noise_gamma",
        type=float,
        default=0.0,
        help="Should we evaluate with additional output noise",
    )
    parser.add_argument(
        "--eval_size", type=int, default=10000, help="Size of valid and test samples"
    )
    parser.add_argument(
        "--eval_noise_type",
        type=str,
        default="additive",
        choices=["additive", "multiplicative"],
        help="Type of noise added at test time",
    )
    parser.add_argument(
        "--eval_noise", type=float, default=0, help="Size of valid and test samples"
    )
    parser.add_argument(
        "--eval_only", type=bool_flag, default=False, help="Only run evaluations"
    )
    parser.add_argument(
        "--eval_from_exp", type=str, default="", help="Path of experiment to use"
    )
    parser.add_argument(
        "--eval_data", type=str, default="", help="Path of data to eval"
    )
    parser.add_argument(
        "--eval_verbose", type=int, default=0, help="Export evaluation details"
    )
    parser.add_argument(
        "--eval_verbose_print",
        type=bool_flag,
        default=False,
        help="Print evaluation details",
    )
    parser.add_argument(
        "--eval_input_length_modulo",
        type=int,
        default=-1,
        help="Compute accuracy for all input lengths modulo X. -1 is equivalent to no ablation",
    )
    
    parser.add_argument("--eval_on_pmlb", type=bool, default=False)
    parser.add_argument("--eval_mcts_on_pmlb", type=bool, default=False)
    parser.add_argument("--eval_in_domain", type=bool, default=False)
    parser.add_argument("--eval_mcts_in_domain", type=bool, default=False)
    
    # debug
    parser.add_argument(
        "--debug_slurm",
        type=bool_flag,
        default=False,
        help="Debug multi-GPU / multi-node within a SLURM job",
    )
    parser.add_argument("--debug", help="Enable all debug flags", action="store_true")

    # CPU / multi-gpu / multi-node
    parser.add_argument("--cpu", type=bool_flag, default=False, help="Run on CPU")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Multi-GPU - Local rank"
    )
    parser.add_argument("--gpu_to_use", type=str, default="0", help="CUDA GPU number to run")
    parser.add_argument(
        "--master_port",
        type=int,
        default=-1,
        help="Master port (for multi-node SLURM jobs)",
    )
    parser.add_argument(
        "--windows",
        type=bool_flag,
        default=False,
        help="Windows version (no multiprocessing for eval)",
    )
    parser.add_argument(
        "--nvidia_apex", type=bool_flag, default=False, help="NVIDIA version of apex"
    )
    parser.add_argument(
        "--max_src_len", type=int, default=200, help="Max Source Length")
    parser.add_argument(
        "--max_target_len", type=int, default=200, help="Max Target Length")
    parser.add_argument(
        "--reward_type", type=str, default='nmse', help="Reward Type")
    parser.add_argument(
        "--reward_coef", type=float, default=1, help="Reward Coeff")
    parser.add_argument(
        "--vf_coef", type=float, default=1e-4, help="PPO vf loss coefficient")
    parser.add_argument(
        "--target_kl", type=float, default=1, help="Target KL for PPO")
    parser.add_argument(
        "--entropy_coef", type=float, default=0.01, help="Coefficient for entropy in PPO Loss")
    parser.add_argument(
        "--kl_regularizer", type=float, default=0.001, help="Coefficient for regularizing KL toward Target KL")
    parser.add_argument(
        "--warmup_epoch", type=int, default=5, help="Warmup period (number of epochs)")
    parser.add_argument(
        "--lr_patience", type=int, default=100, help="LRonPlateau patience")

    parser.add_argument(
        "--save_model", type=bool, default=True, help="Should we save model?")
    parser.add_argument(
        "--save_eval_dic", type=str, default='./eval_result', help="directory to save evaluation scores")
    parser.add_argument(
        "--update_modules", type=str, default="all", help="Ablation of Updating Model:all,dec,enc-dec")
    parser.add_argument(
        "--actor_lr", type=float, default=1e-6, help="Actor Learning Rate")
    parser.add_argument(
        "--critic_lr", type=float, default=1e-5, help="Critic Learning Rate")
    parser.add_argument(
        "--kl_coef", type=float, default=0.01, help="KL Coef")
    parser.add_argument(
        "--rl_alg", type=str, default="ppo", help="RL Algorithm: ppo, reinforce")
    
    parser.add_argument(
        "--run_id",
        type=int,
        default=1,
        help="Run ID",)
    
    parser.add_argument(
        "--pmlb_data_type",
        type=str,
        default="feynman",
        help="pmlb dataset type",
    )
    
    parser.add_argument(
        "--target_noise",
        type=float,
        default=0.0,
        help="targte noise for the pmlb added to the y_to_fit",
    )
    
    parser.add_argument(
            "--prediction_sigmas",
            type=str,
            default="1,2,4,8,16",
            help="sigmas value for generation predicts",
        )
    
    
    
    return parser
