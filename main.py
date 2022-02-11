import torch
import pandas as pd
import os
import primary_networks
import sub_networks
import config_0
import config_1
# from config import ConfigMain
# from config import ConfigPrimary
# from config import ConfigSubModel
import utils
import argparse
import random
import numpy as np
import traceback


def run_task(config, config_primary, config_subs, iter_num, task_type):
    task_name_list, max_len, text_col,\
        model_name, embeddings_version, \
        embeddings_path, label_col, key_col, submodels_list = \
        utils.read_config_main(config)
    for task_name in task_name_list:
        models_path = ''.join(('trained_models//', task_name, '//',
                               embeddings_version, '//', str(iter_num), '//'))
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        # utils.check_input(model_name, task_type)
        train_df = utils.read_df(task_name, 'train')
        val_df = utils.read_df(task_name, 'val')
        test_df = utils.read_df(task_name, 'test')

        # Set seed
        random.seed(iter_num)
        np.random.seed(iter_num)
        torch.manual_seed(iter_num)
        torch.cuda.manual_seed_all(iter_num)

        if task_type == 'train_primary':
            try:
                primary_networks.run_primary(task_name, model_name, train_df, val_df, test_df,
                                             max_len, text_col, embeddings_version,
                                             embeddings_path, config_primary, models_path,
                                             label_col, key_col, iter_num)
            except:
                traceback.print_exc()
        else:
            # todo: Change it
            pass
    return
    # elif task_type == "train_sub":
    #     sub_networks.train_sub_models(task_name, model_name, train_df, val_df, test_df, max_len,
    #                                   text_col, embeddings_version, embeddings_path,
    #                                   config_subs, models_path, label_col,
    #                                   key_col, submodels_list, sub_nn, iter_num)


# def create_models_df():
#     models_dict = {'primary': {}, 'over_under': {}, 'model_12': {},
#                    'model_13': {}, 'model_14': {}, 'model_15': {},
#                    'model_23': {}, 'model_24': {}, 'model_25': {},
#                    'model_34': {}, 'model_35': {},'model_45': {}}
#     columns_df = ['hid_dim_lstm', 'dropout', 'lin_output_dim', 'lr',
#                   'epochs_num', 'batch_size', 'momentum', 'accuracy']
#     for model_name in models_dict.keys():
#         models_dict[model_name] = ['' for col_num in range(len(columns_df))]
#     models_df = pd.DataFrame.from_dict(models_dict, orient='index')
#     models_df.columns = columns_df
#     return models_df


def main(iter_num, task_type, cuda_id):
    config = config_0.ConfigMain() if cuda_id == 0 else config_1.ConfigMain()
    config_primary = config_0.ConfigPrimary() if cuda_id == 0 else config_1.ConfigPrimary()
    config_subs = config_0.ConfigSubModel() if cuda_id == 0 else config_1.ConfigSubModel()
    # models_df = create_models_df()
    run_task(config, config_primary, config_subs, iter_num, task_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_num", type=int, default=1)
    parser.add_argument("--task_type", type=str, default="train_primary")
    parser.add_argument("--cuda_id", type=str, default=0)
    # parser.add_argument("--sub_nn", type=str, default="regular")

    hp = parser.parse_args()
    cuda_id = int(hp.cuda_id)  # We use single GPU
    print(f"Running Using CUDA {cuda_id}")
    main(hp.iter_num, hp.task_type, cuda_id)
