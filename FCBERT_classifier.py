import torch
import torch.nn as nn
from torch.nn import functional as Func
from transformers import BertModel, RobertaModel, AlbertModel, DistilBertModel
from utils import create_prox_mat, get_loss, create_mass_encoder, create_prox_dom
import torch.nn.functional as f


def get_model(embeddings_path):
    if embeddings_path.startswith("bert-"):
        return BertModel.from_pretrained(embeddings_path)
    elif embeddings_path.startswith("roberta-"):
        return RobertaModel.from_pretrained(embeddings_path)
    elif embeddings_path.startswith("albert-"):
        return AlbertModel.from_pretrained(embeddings_path)
    elif embeddings_path.startswith("distilbert-"):
        return DistilBertModel.from_pretrained(embeddings_path)


class FCBERT_PRIMARY(nn.Module):
    def __init__(self, embeddings_path, labels_num, iter_num,
                 loss_type='OCE', dist_dict=None,
                 denominator=None, alpha=None, beta=None):
        super(FCBERT_PRIMARY, self).__init__()
        torch.manual_seed(iter_num)
        torch.cuda.manual_seed_all(iter_num)
        self.embeddings_path = embeddings_path
        self.model = get_model(embeddings_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_pred = nn.Linear(self.model.config.hidden_size, labels_num)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.loss_type = loss_type
        self.labels_num = labels_num
        self.dist_dict = dist_dict
        self.alpha = alpha
        self.inv_prox_mat = torch.tensor(create_prox_mat(dist_dict, True)).to(self.device) \
            if dist_dict is not None else None
        self.prox_mat = torch.tensor(create_prox_mat(dist_dict, False)).to(self.device) \
            if dist_dict is not None else None
        self.mass_encoder = create_mass_encoder(labels_num).to(self.device)
        self.prox_dom = create_prox_dom(self.prox_mat).to(self.device)
        self.norm_prox_mat = f.normalize(self.prox_mat, p=1, dim=0)
        # self.norm_inv_prox_mat = self.inv_prox_mat / torch.min(self.inv_prox_mat) if dist_dict is not None else None
        # self.se_tensor = torch.tensor([[(i - true_label)**2 for i in range(labels_num)]
        #                                for true_label in range(labels_num)])
        # self.alpha = alpha
        # self.beta = beta

    def forward(self, input_ids, attention_mask, ground_truth):
        if self.embeddings_path.startswith("distilbert-"):
            pooler = self.model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0, :]
        else:
            pooler = self.model(input_ids=input_ids, attention_mask=attention_mask)[1]
        Y1 = self.linear_pred(pooler)
        lsm = self.logsoftmax(Y1)
        softmax_values = Func.softmax(Y1, dim=1)
        _, predictions = softmax_values.max(1)
        if lsm.shape[0] > input_ids.shape[0]:
            lsm = lsm[:input_ids.shape[0]]
        loss_val = get_loss(self.loss_type, lsm, softmax_values, ground_truth,
                            self.device, self.prox_dom, self.prox_mat, self.inv_prox_mat,
                            self.norm_prox_mat, self.mass_encoder, self.labels_num, self.alpha)
        del input_ids, attention_mask, pooler, Y1, lsm
        torch.cuda.empty_cache()
        return loss_val, predictions, softmax_values


class FCBERT_REGRESSION(nn.Module):
    def __init__(self, embeddings_path, labels_num, iter_num):
        super(FCBERT_REGRESSION, self).__init__()
        torch.manual_seed(iter_num)  # torch.manual_seed(12345)
        self.model = get_model(embeddings_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_pred = nn.Linear(self.model.config.hidden_size, 1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.labels_num = labels_num

    def forward(self, input_ids, attention_mask, ground_truth):
        pooler = self.model(input_ids=input_ids, attention_mask=attention_mask)[1]
        predictions = self.linear_pred(pooler)
        loss_val = torch.mean((predictions - ground_truth) ** 2)
        rounded_preds = torch.tensor([round(pred.item()) for pred in predictions]).to(self.device)
        del input_ids, attention_mask, pooler
        torch.cuda.empty_cache()
        return loss_val, rounded_preds, torch.zeros((len(rounded_preds), self.labels_num))


class FCBERT_SUB(nn.Module):
    def __init__(self, embeddings_path, labels_num, iter_num):
        super(FCBERT_SUB, self).__init__()
        torch.manual_seed(iter_num)  # torch.manual_seed(12345)
        self.model = BertModel.from_pretrained(embeddings_path)
        self.loss_dist = nn.MSELoss()
        self.loss_flag = nn.NLLLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_pred = nn.Linear(self.model.config.hidden_size, labels_num)
        self.linear_flag = nn.Linear(self.model.config.hidden_size, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask, labels_dist, flags):
        pooler = self.model(input_ids=input_ids, attention_mask=attention_mask)[1]
        Y1 = self.linear_pred(pooler)
        lsm_dist = self.logsoftmax(Y1)
        softmax_values_dist = Func.softmax(Y1, dim=1)
        # _, predictions_by_dist = softmax_values_dist.max(1)
        # if lsm_dist.shape[0] > input_ids.shape[0]:
        #     lsm_dist = lsm_dist[:input_ids.shape[0]]
        if softmax_values_dist.shape[0] > input_ids.shape[0]:
            softmax_values_dist = softmax_values_dist[:input_ids.shape[0]]
        loss_dist = self.loss_dist(softmax_values_dist, labels_dist)

        Y2 = self.linear_flag(pooler)
        lsm_flag = self.logsoftmax(Y2)
        softmax_values_flag = Func.softmax(Y2, dim=1)
        _, predictions_flag = softmax_values_flag.max(1)
        if lsm_flag.shape[0] > input_ids.shape[0]:
            lsm_flag = lsm_flag[:input_ids.shape[0]]
        loss_flag = self.loss_flag(lsm_flag, flags.to(self.device))
        loss_val = 0.5*loss_dist + 0.5*loss_flag
        return loss_val, softmax_values_dist, predictions_flag, loss_dist
