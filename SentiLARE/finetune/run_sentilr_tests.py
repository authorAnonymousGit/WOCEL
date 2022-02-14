import os
import numpy as np

SEEDS = [1, 2, 3, 4, 5]
LOSS_TYPES = ["OCE_loss_final_with_prox_dynamic_L7_with_alpha", "CrossEntropy"]
ALPHAS = np.linspace(0.5, 0.9, 5)  # Irrelevant for CrossEntropy

# SEEDS = [1, 3, 5]
# LOSS_TYPES = ['OCE_loss_final_with_prox_dynamic_L7_with_alpha_ablation_prox',
#               'OCE_loss_final_with_prox_dynamic_L7_with_alpha_ablation_conf']
# ALPHAS = [0.9]  # Irrelevant for CrossEntropy

# TODO: Consider change per_gpu_train_batch_size, num_train_epochs
main_output_dir = "../sent_finetune/sst/"
running_command = "CUDA_VISIBLE_DEVICES=1 python run_sent_sentilr_roberta.py \
                                              --data_dir ../data/sent/sst \
                                              --model_type roberta \
                                              --model_name_or_path ../pretrain_model/ \
                                              --task_name sst \
                                              --do_train \
                                              --do_eval \
                                              --max_seq_length 512 \
                                              --per_gpu_train_batch_size 16 \
                                              --learning_rate 2e-5 \
                                              --num_train_epochs 6 \
                                              --output_dir {output_dir} \
                                              --logging_steps 100 \
                                              --save_steps 100 \
                                              --warmup_steps 100 \
                                              --eval_all_checkpoints \
                                              --overwrite_output_dir \
                                              --seed {seed} \
                                              --loss_type {loss_type} \
                                              --alpha {alpha}"

os.system("export NVIDIA_VISIBLE_DEVICES='1'")
os.system("export CUDA_VISIBLE_DEVICES='1'")

for seed in SEEDS:
    print("*" * 20, f"SEED {seed}", "*" * 20)
    for loss_type in LOSS_TYPES:
        if loss_type == 'CrossEntropy':
            print('~' * 10, "RUNNING SentiLARE, ", loss_type, '~' * 10)
            output_dir = main_output_dir + str(seed) + '/' + loss_type + '/'
            os.system(running_command.format(output_dir=output_dir, seed=seed,
                                             loss_type=loss_type, alpha=0.0))
        else:  # OCE loss
            for alpha in ALPHAS:
                print('~' * 10, "RUNNING SentiLARE, ", loss_type, f" With alpha {alpha}", '~' * 10)
                output_dir = main_output_dir + str(seed) + '/' + loss_type + '/alpha_' + str(int(10 * alpha)) + '/'
                os.system(running_command.format(output_dir=output_dir, seed=seed,
                                                 loss_type=loss_type, alpha=alpha))
