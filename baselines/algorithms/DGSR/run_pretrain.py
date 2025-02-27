# This file is used to pre-train the DGSR model for a specific dataset that uses a specific number of input variables (covariates) as defined by the test dataset equation selected.
# This will train the model essentially forever, and continually saves the model during training. After a period of 4-5 hours, the model performance loss will plateau.
# The user will need to then manually stop the training process, and copy the saved model ('controller.pt') and model configuration file ('config.json') from the log directory ('./log/{$RUN}', where $RUN is a folder name that includes the trained dataset and date as a string) to the pre-trained models directory (./models/dgsr_pre-train/), and then update the path to load when testing in 'run_recovery_multi.py' in the dict 'COVARS_TO_PRE_TRAINED_MODEL'.

import logging
import os
import time

from exp_main import top_main

PRIORITY_QUEUE_TRAINING = False
GP_MELD = False
TEST_DATASET = "fn_d_all_10"

def main(test_dataset, gp_meld, priority_queue_training):
    result = top_main(  # noqa: F841  # pylint: disable=unused-variable
        test_dataset=test_dataset,
        training_equations=10000,
        training_epochs=1000000,
        batch_outer_datasets=24,
        batch_inner_equations=300,
        pre_train=True,
        load_pre_trained_path="",
        priority_queue_training=priority_queue_training,
        gp_meld=gp_meld,
        model="TransformerTreeEncoderController",
        train_path="",
        risk_seeking_pg_train=True,  # False = CE loss train
        data_gen_max_len=20,  # set to 30 for generating 5 covar dataset
        data_gen_max_ops=5,
        data_gen_equal_prob_independent_vars=True,
        data_gen_remap_independent_vars_to_monotic=True,
        data_gen_force_all_independent_present=True,
        # data_gen_operators="add:10,mul:10,sub:5,div:5,pow2:4,pow3:2,pow4:1,pow5:1,ln:10,exp:4,sin:4,cos:4",  # Koza
        data_gen_operators="add:10,mul:10,sub:5,div:5,pow2:4,pow3:2,pow4:1,pow5:1,ln:10,exp:4,sin:4,cos:4,sign:4",  # Koza-sign
        # data_gen_operators="add:10,mul:10,sub:5,div:5,pow2:4,pow3:2,pow4:1,pow5:1,ln:10,exp:4,sin:4,cos:4,sign:4,tanh:4,cosh:4,abs:4",  # Koza-chaotic
        # data_gen_operators="add:10,mul:10,sub:5,div:5,pow2:4,pow3:2,pow4:1,pow5:1,ln:10,exp:4,sin:4,cos:4,cosh:4,tanh:4",  # Koza-turbulence
        data_gen_lower_nbs_ops=3,
        learning_rate=0.001,
        data_gen_create_eqs_with_constants= True,
    )

if __name__ == "__main__":
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    test_dataset = TEST_DATASET
    gp_meld = GP_MELD
    priority_queue_training = PRIORITY_QUEUE_TRAINING

    file_name = os.path.basename(os.path.realpath(__file__)).split(".py")[0]
    path_run_name = "{}-{}".format(file_name, time.strftime("%Y%m%d-%H%M%S"))
    logging.basicConfig(
        filename="./logs/{}_log.txt".format(path_run_name),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    main(test_dataset=test_dataset, gp_meld=gp_meld, priority_queue_training=priority_queue_training)
