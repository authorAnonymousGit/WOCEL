import utils
import time
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from FCBERT_classifier import FCBERT_PRIMARY, FCBERT_REGRESSION
import csv
import copy
from inference import results_analysis
import os


def get_model(model_name, embeddings_path, labels_num, loss_type, dist_dict, denominator,
              iter_num, alpha, beta):
    if model_name == "FCBERT":
        return FCBERT_PRIMARY(embeddings_path, labels_num=labels_num, iter_num=iter_num,
                              loss_type=loss_type, dist_dict=dist_dict,
                              denominator=denominator, alpha=alpha, beta=beta)
    elif model_name == "FCBERT_REGRESSION":
        return FCBERT_REGRESSION(embeddings_path, labels_num, iter_num)
    else:
        raise TypeError("The model " + model_name + " is not defined")


def train_primary_model(train_df, val_df, test_df, text_feature,
                        embeddings_version, max_len, batch_size, label_col,
                        key_col, model_name, embeddings_path, labels_num,
                        loss_type, device, epochs_num, models_path, lr, iter_num, model_selection_procedure,
                        alpha=None, beta=None):
    model_file_name = 'primary_' + loss_type
    torch.manual_seed(iter_num)
    if alpha is not None:
        model_file_name += '_alpha_' + str(int(alpha * 10))
        if beta is not None:
            model_file_name += '_beta_' + str(int(beta * 10))

    print("model_file_name: ")
    print(model_file_name)

    model_summary_rows = []

    train_dataloader, val_dataloader = utils.create_dataloaders_train(train_df, val_df,
                                                                      text_feature,
                                                                      embeddings_version,
                                                                      max_len, batch_size,
                                                                      label_col, key_col)

    dist_dict_train, dist_dict_val, dist_dict_test = train_df[label_col].value_counts().to_dict(), \
                                                    val_df[label_col].value_counts().to_dict(), \
                                                    test_df[label_col].value_counts().to_dict()
    denominator_train, denominator_val, denominator_test = len(train_df), len(val_df), len(test_df)
    prox_mat_train = torch.tensor(utils.create_prox_mat(dist_dict_train, inv=False)).to(device)
    prox_mat_val = torch.tensor(utils.create_prox_mat(dist_dict_val, inv=False)).to(device)
    prox_mat_test = torch.tensor(utils.create_prox_mat(dist_dict_test, inv=False)).to(device)

    model = get_model(model_name, embeddings_path, labels_num, loss_type,
                      dist_dict_train, denominator_train, iter_num, alpha=alpha, beta=beta)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    train_len, val_len = len(train_df), len(val_df)
    train_accuracy_list, val_accuracy_list, test_accuracy_list = [], [], []
    train_cem_list, val_cem_list, test_cem_list = [], [], []
    train_mae_list, val_mae_list, test_mae_list = [], [], []
    best_val_metric = 0.0
    total_steps = len(train_dataloader) * epochs_num
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    best_model = None
    for epoch in range(epochs_num):
        utils.print_epochs_progress(epoch, epochs_num)
        start_train = time.process_time()
        acc_train = 0.0
        loss_scalar = 0.0
        cem_num_train = 0.0
        cem_den_train = 0.0
        mae_train = 0.0
        model.train()
        optimizer.zero_grad()
        t0 = time.time()
        predictions_dict_train = dict()
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx % 10 == 0 and not batch_idx == 0:
                utils.print_batches_progress(t0, batch_idx, train_dataloader)
            input_ids = batch[0].to(device, dtype=torch.long)
            masks = batch[1].to(device, dtype=torch.long)
            labels = batch[2].to(device, dtype=torch.long)
            key_ids = batch[3]
            model.zero_grad()
            loss, predictions, probabilities = model(input_ids, masks, labels)
            loss_scalar += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            acc_train += torch.sum(predictions == labels)
            cem_num_train += sum(prox_mat_train[predictions, labels])
            cem_den_train += sum(prox_mat_train[labels, labels])
            mae_train += torch.sum(torch.abs(labels - predictions))

            optimizer.step()
            scheduler.step()
            predictions_dict_train = utils.update_predictions_dict('primary', predictions_dict_train,
                                                                   key_ids,
                                                                   labels,
                                                                   probabilities,
                                                                   predictions)
            del input_ids, masks, labels, loss, predictions, probabilities
            torch.cuda.empty_cache()

        loss_scalar /= (batch_idx + 1)
        acc_train /= train_len
        cem_train = cem_num_train / cem_den_train
        mae_train /= train_len

        acc_train = acc_train.item()
        cem_train = cem_train.item()
        mae_train = mae_train.item()

        end_train = time.process_time()
        total_time = end_train - start_train
        utils.print_train_epoch(epoch, acc_train, train_len, loss_scalar, total_time)

        utils.print_train_epoch_end(t0)
        val_acc, val_mae, val_cem, predictions_dict_val = evaluate_val(model, val_dataloader,
                                                                       device, val_len, prox_mat_val)
        test_acc, test_mae, test_cem, predictions_dict_test = evaluate_test(test_df, model, device,
                                                                            embeddings_version,
                                                                            max_len, text_feature,
                                                                            batch_size, label_col, key_col,
                                                                            prox_mat_test)
        train_accuracy_list.append(round(float(acc_train), 5))
        val_accuracy_list.append(round(float(val_acc), 5))
        test_accuracy_list.append(round(float(test_acc), 5))
        train_cem_list.append(round(float(cem_train), 5))
        val_cem_list.append(round(float(val_cem), 5))
        test_cem_list.append(round(float(test_cem), 5))
        train_mae_list.append(round(float(mae_train), 5))
        val_mae_list.append(round(float(val_mae), 5))
        test_mae_list.append(round(float(test_mae), 5))

        # loss_list_val.append(val_loss)
        is_best_model = False
        if model_selection_procedure == 'cem' and val_cem > best_val_metric:
            is_best_model = True
            best_val_metric = val_cem
        elif model_selection_procedure == 'acc' and val_acc > best_val_metric:
            is_best_model = True
            best_val_metric = val_acc
        if is_best_model:
            torch.save(model.state_dict(), models_path + model_file_name + '.pkl')
            best_model = copy.deepcopy(model)
        utils.save_predictions_to_df(predictions_dict_train, models_path, 'train',
                                     model_file_name + "epoch" + str(epoch))
        utils.save_predictions_to_df(predictions_dict_val, models_path, 'validation',
                                     model_file_name + "epoch" + str(epoch))
        utils.save_predictions_to_df(predictions_dict_test, models_path, 'test',
                                     model_file_name + "epoch" + str(epoch))

        model_summary_rows.extend([[model_file_name, epoch + 1, acc_train, mae_train, cem_train,
                                   val_acc, val_mae, val_cem, test_acc, test_mae, test_cem]])

    # utils.save_model(models_df, 'primary', labels_num, loss_type, lr,
    #                  epochs_num, batch_size, best_val_acc)

    utils.print_summary(models_path, model_file_name, "train", "accuracy", train_accuracy_list)
    utils.print_summary(models_path, model_file_name, "train", "cem", train_cem_list)
    utils.print_summary(models_path, model_file_name, "train", "mae", train_mae_list)

    utils.print_summary(models_path, model_file_name, "validation", "accuracy", val_accuracy_list)
    utils.print_summary(models_path, model_file_name, "validation", "cem", val_cem_list)
    utils.print_summary(models_path, model_file_name, "validation", "mae", val_mae_list)

    utils.print_summary(models_path, model_file_name, "test", "accuracy", test_accuracy_list)
    utils.print_summary(models_path, model_file_name, "test", "cem", test_cem_list)
    utils.print_summary(models_path, model_file_name, "test", "mae", test_mae_list)

    test_acc, _, _, predictions_dict_test = evaluate_test(test_df, best_model, device, embeddings_version,
                                                          max_len, text_feature, batch_size, label_col,
                                                          key_col, prox_mat_test)
    utils.save_predictions_to_df(predictions_dict_test, models_path, 'test', model_file_name)
    utils.print_test_results(models_path, model_file_name, test_acc)
    return model_file_name, model_summary_rows


def run_primary(task_name, model_name, train_df, val_df, test_df,
                max_len, text_feature, embeddings_version,
                embeddings_path, config_primary, models_path,
                label_col, key_col, iter_num):
    device = utils.find_device()

    loss_types, alphas, betas, labels_num, epochs_num, lr, batch_size, model_selection_procedure = \
        utils.read_config_networks(config_primary, "train_primary")

    primary_summary_rows = [['model_file_name', 'epoch',
                             'train_acc', 'train_mae', 'train_cem',
                             'val_acc', 'val_mae', 'val_cem',
                             'test_acc', 'test_mae', 'test_cem']]
    orig_models_path = models_path
    for loss_type in loss_types:
        # if loss_type == 'CrossEntropy':
        #     models_path = orig_models_path + "CrossEntropy" + '//'
        # else:
        #     models_path = orig_models_path + loss_type[-1] + '//'
        models_path = orig_models_path + loss_type + '//'
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        if loss_type == 'CrossEntropy':
            print('~' * 10, "primary", loss_type, '~' * 10)
            model_file_name, model_summary_rows = train_primary_model(train_df, val_df, test_df,
                                                                      text_feature, embeddings_version, max_len,
                                                                      batch_size, label_col, key_col, model_name,
                                                                      embeddings_path, labels_num, loss_type,
                                                                      device, epochs_num, models_path, lr,
                                                                      iter_num, model_selection_procedure)
            # val_acc, val_mae, val_cem = results_analysis.evaluate_results(models_path + 'predictions_validation_' +
            #                                                         model_file_name + '.csv',
            #                                                         'label', 'prediction')
            # test_acc, test_mae, test_cem = results_analysis.evaluate_results(models_path + 'predictions_test_' +
            #                                                           model_file_name + '.csv',
            #                                                           'label', 'prediction')
            primary_summary_rows.extend(model_summary_rows)

        else:  # loss_type starts with 'OrdinalTextClassification'
            for alpha in alphas:
                if loss_type != 'OrdinalTextClassification-A':
                    betas = [None, ]
                for beta in betas:
                    if beta is not None:
                        if alpha + beta > 1:
                            continue
                    print('~' * 10, "primary, Loss type: ", loss_type, " alpha: ", alpha,
                          " beta: ", beta, '~' * 10)
                    model_file_name, model_summary_rows = train_primary_model(train_df, val_df, test_df,
                                                                              text_feature, embeddings_version,
                                                                              max_len, batch_size, label_col,
                                                                              key_col, model_name,
                                                                              embeddings_path, labels_num, loss_type,
                                                                              device, epochs_num, models_path, lr,
                                                                              iter_num, model_selection_procedure,
                                                                              alpha, beta)

                    # val_acc, val_mae, val_cem = results_analysis.evaluate_results(models_path +
                    #                                                               'predictions_validation_' +
                    #                                                               model_file_name + '.csv',
                    #                                                               'label', 'prediction')
                    # test_acc, test_mae, test_cem = results_analysis.evaluate_results(models_path +
                    #                                                                  'predictions_test_' +
                    #                                                                  model_file_name + '.csv',
                    #                                                                  'label', 'prediction')
                    primary_summary_rows.extend(model_summary_rows)

    with open(orig_models_path + "primary_results_summary.csv", "a+") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerows(primary_summary_rows)
    return


def evaluate_val(model, val_dataloader, device, val_len, prox_mat):
    start_val = time.process_time()
    acc_num_val = 0.0
    loss_scalar = 0.0
    cem_num = 0.0
    cem_den = 0.0
    mae = 0.0
    model.eval()
    predictions_dict_val = dict()
    for batch_idx, batch in enumerate(val_dataloader):
        input_ids = batch[0].to(device, dtype=torch.long)
        masks = batch[1].to(device, dtype=torch.long)
        labels = batch[2].to(device, dtype=torch.long)
        key_ids = batch[3]
        with torch.no_grad():
            loss, predictions, probabilities = model(input_ids, masks, labels)
            predictions_dict_val = utils.update_predictions_dict('primary', predictions_dict_val,
                                                                 key_ids,
                                                                 labels,
                                                                 probabilities,
                                                                 predictions)
            loss_scalar += loss.item()
            # acc_num_val += utils.add_correct_num(predictions, labels)
            cem_num += sum(prox_mat[predictions, labels])
            cem_den += sum(prox_mat[labels, labels])
            acc_num_val += torch.sum(predictions == labels)
            mae += torch.sum(torch.abs(labels - predictions))
        del input_ids, masks, labels, loss, predictions, probabilities
        torch.cuda.empty_cache()
    cem = cem_num / cem_den
    loss_scalar /= val_len
    acc_num_val /= val_len
    mae /= val_len
    end_val = time.process_time()
    total_time = end_val - start_val
    utils.print_validation_epoch(acc_num_val, val_len, loss_scalar, total_time)
    return float(acc_num_val), float(mae), float(cem), predictions_dict_val


def evaluate_test(test_df, model, device, embeddings_version, max_len,
                  text_feature, batch_size, label_col, key_col, prox_mat):
    start_val = time.process_time()
    test_dataloader = utils.create_dataloaders_test(test_df, embeddings_version,
                                                    max_len, text_feature, label_col,
                                                    key_col, batch_size)
    test_len = len(test_df)
    acc = 0.0
    cem_num = 0.0
    cem_den = 0.0
    mae = 0.0
    model.eval()
    predictions_dict_test = dict()
    for batch_idx, batch in enumerate(test_dataloader):
        input_ids = batch[0].to(device, dtype=torch.long)
        masks = batch[1].to(device, dtype=torch.long)
        labels = batch[2].to(device, dtype=torch.long)
        key_ids = batch[3]
        with torch.no_grad():
            # start_time = time.time()
            _, predictions, probabilities = model(input_ids, masks, labels)
            # end_time = time.time()
            # print("Total test time: ", end_time - start_time)
            predictions_dict_test = utils.update_predictions_dict('primary', predictions_dict_test,
                                                                  key_ids, labels,
                                                                  probabilities,
                                                                  predictions)
            # acc_num_test += utils.add_correct_num(predictions, labels)
            acc += torch.sum(predictions == labels)
            mae += torch.sum(torch.abs(labels - predictions))
            cem_num += sum(prox_mat[predictions, labels])
            cem_den += sum(prox_mat[labels, labels])

        del input_ids, masks, labels, predictions, probabilities
        torch.cuda.empty_cache()

    cem = cem_num / cem_den
    acc /= test_len
    mae /= test_len

    end_val = time.process_time()
    total_time = end_val - start_val
    print("total time: ", total_time)
    return float(acc), float(mae), float(cem), predictions_dict_test
