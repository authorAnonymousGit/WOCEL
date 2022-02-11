import numpy as np


class ConfigMain:
    # AMAZON_DATASETS = ['AmazonAll_Beauty', 'AmazonSoftware', 'AmazonFashion', 'AmazonMovies_and_TV', 'AmazonElectronics']
    # AMAZON_DATASETS = ['sst-5']
    # Only for Amazon. Otherwise mark it as a comment
    # TASK_NAME = [dataset + str(df_idx) for dataset in AMAZON_DATASETS for df_idx in range(0, 1)]
    TASK_NAME = ['sst-5']   #  / 'SemEval2017' / 'AmazonFashion'

    MAX_LEN = 512
    TEXT_FEATURE = 'text'
    MODEL_NAME = 'FCBERT'
    EMBEDDINGS_VERSION = 'roberta-base'  # "bert-base-uncased"/"bert-large-uncased"/"roberta-base"/"albert-large-v2"/"distilbert-base-uncased"
    EMBEDDINGS_PATH = 'roberta-base'  # "bert-base-uncased"/"bert-large-uncased"/"roberta-base"/"albert-large-v2"/"distilbert-base-uncased"
    LABEL_FEATURE = "overall"
    KEY_FEATURE = "key_index"
    SUB_MODELS = ['model_12', 'model_13', 'model_14', 'model_15',
                  'model_23', 'model_24', 'model_25', 'model_34',
                  'model_35', 'model_45']
    # SUB_MODELS = ['model_12', 'model_13', 'model_23']
    # 5 For 'sst-5' and 'AmazonFashion', 3 For 'SemEval2017' (['model_12', 'model_13', 'model_23'])


class ConfigPrimary:
    # Try different variations of loss functions out of "CrossEntropy"/ "OrdinalTextClassification-?".
    # LOSS_TYPE = ["OCE_loss_final_with_prox_dynamic_L4",
    #              "OCE_loss_final_with_prox_dynamic_L5",
    #              "OCE_loss_final_with_prox_dynamic_L6",
    #              "OCE_loss_final_with_prox_dynamic_L7",
    #              "CrossEntropy"]
    # LOSS_TYPE = ["OCE_loss_final_with_prox_dynamic_L7_with_alpha", "CrossEntropy"]
    LOSS_TYPE = ["OCE_loss_final_with_prox_dynamic_L7_with_alpha_ablation_conf",
                 "OCE_loss_final_with_prox_dynamic_L7_with_alpha_ablation_prox"]
    ALPHA = [0.6, ]  # np.linspace(0.5, 0.9, 5)  # Irrelevant
    BETA = np.linspace(0.0, 0.0, 1)  # Irrelevant
    LABELS_NUM = 5  # 3 For SemEval2017, 5 for sst-5 and 'AmazonFashion'
    EPOCHS_NUM = 6
    LEARNING_RATE = [0.06, 0.15, 10]  # Irrelevant
    BATCH_SIZE = [16, 16, 1]
    MODEL_SELECTION_PROCEDURE = 'cem'  # acc / cem


class ConfigSubModel:
    LOSS_TYPE = "CrossEntropy"
    LABELS_NUM = 2
    EPOCHS_NUM = 5
    LEARNING_RATE = [0.01, 0.15, 15]  # Irrelevant
    BATCH_SIZE = [16, 16, 1]


class ConfigGraphClassification:
    LABELS_NUM = 5  # 3 For SemEval2017, 5 for sst-5 and AmazonFashion
    SINGLE_GRAPH_STRUCTURE = False # Irrelevant
    EDGE_CREATION_PROCEDURE = ['pyramid_narrow_disconn_dist_1', 'pyramid_narrow_conn_dist_1',
                               'pyramid_wide_disconn_dist_1', 'pyramid_wide_conn_dist_1']
    # 'star', 'star_conn_kids', 'complete' are also available but we did not use them in out experiments

    # Select the method of yielding the GCN's prediction
    FINAL_PREDICTION_PROCEDURE = ['primary', ]  # 'primary' / 'avg_all'

    MODEL_DATA_PROCEDURE = ['by_origin_pred', ]  # 'by_origin_pred' / 'all'. We used only by_origin_pred.

    # Try different variations of loss functions out of "CrossEntropy"/ "OrdinalTextClassification-?".
    LOSS_TYPE = ["CrossEntropy", "OrdinalTextClassification-E",
                 "OrdinalTextClassification-F", "OrdinalTextClassification-G",
                 "OrdinalTextClassification-H", "OrdinalTextClassification-I",
                 "OrdinalTextClassification-J", "OrdinalTextClassification-K"]

    ALPHA = np.linspace(0.0, 0.0, 1)  # Irrelevant
    BIDIRECTIONAL = True
    HIDDEN_CHANNELS = 64
    BATCH_SIZE = 32
    NODE_FEATURES_NUM = 5  # 3 For SemEval2017, 5 for sst-5 and 'AmazonFashion'
    EPOCHS_NUM = 5

    # Select the primary model by its loss function. The GCN will be built upon this model.
    # The possible options are "primary_CrossEntropy"/ "primary_OrdinalTextClassification-?".
    # The used model has to be fine-tuned in advance as a primary model.
    PRIMARY_MODEL_NAME_BASELINE = 'primary_CrossEntropy'
    PRIMARY_MODEL_NAME_GNN = PRIMARY_MODEL_NAME_BASELINE
    PRIMARY_LOSS_TYPE = PRIMARY_MODEL_NAME_BASELINE.split('_')[1]
    MODEL_SELECTION_PROCEDURE = ['val_cem', ]  # 'test_acc'/'each_epoch'/'val_loss'/'val_cem'/'val_acc'

