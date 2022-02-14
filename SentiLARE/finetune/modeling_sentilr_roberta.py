# PyTorch RoBERTa model.
# This code is modified based on modeling_roberta.py in pytorch-transformers.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss

from modeling_sentilr import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu
from pytorch_transformers import RobertaConfig

# New Imports
import numpy as np
import torch.nn.functional as f
from torch.nn import functional as Func


logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}


#################### OUR LOSS ####################

def create_prox_mat(dist_dict, inv=True):
    labels = {int(label) for label in dist_dict.keys()}
    denominator = sum(dist_dict.values())
    prox_mat = np.zeros([len(labels), len(labels)])
    for label1 in labels:
        for label2 in labels:
            minlabel, maxlabel = min(label1, label2), max(label1, label2)
            numerator = dist_dict[label1] / 2
            if minlabel == label1:  # Above the diagonal
                for tmp_label in range(minlabel + 1, maxlabel + 1):
                    numerator += dist_dict[tmp_label]
            else:  # Under the diagonal
                for tmp_label in range(maxlabel - 1, minlabel - 1, -1):
                    numerator += dist_dict[tmp_label]
            if inv:
                prox_mat[label1 - 1][label2 - 1] = (-np.log(numerator / denominator))**-1
            else:
                prox_mat[label1 - 1][label2 - 1] = -np.log(numerator / denominator)
    return torch.tensor(prox_mat)


def create_prox_dom(prox_mat):
    labels_num = prox_mat.shape[0]
    prox_dom_mat = []
    for label in range(prox_mat.shape[0]):
        label_prox_ordered = reversed(prox_mat[:,label].argsort()).tolist()
        prox_dom_mat.append(torch.tensor([[0.0 if label_prox_ordered.index(ind_axe) <
                                           label_prox_ordered.index(ind_inner) else 1.0
                                           for ind_inner in range(labels_num)]
                                          for ind_axe in range(labels_num)]))
    return torch.stack(prox_dom_mat)


def create_mass_encoder(labels_num):
    mass_encoder = []
    for label in range(labels_num):
        mass_encoder.append(torch.tensor([[1.0 if ((ind == label and ind == prediction) or
                                                   (prediction > label and label < ind <= prediction) or
                                                   (prediction < label and label > ind >= prediction))
                                           else 0.0 for ind in range(labels_num)]
                                          for prediction in range(labels_num)]).T)

    return torch.stack(mass_encoder)


def OCE_loss_final_with_prox_dynamic_L7_with_alpha(softmax_vals, targets, prox_dom, norm_prox_mat,
                                                   labels_num, alpha, device):
    num_examples = targets.shape[0]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    elem_weight = torch.tensor(norm_prox_mat[:, targets]).t()
    wrong_mass_weight = torch.sum(one_hot_target_comp * softmax_vals, dim=1).unsqueeze(dim=1) * one_hot_target_comp
    mass_weights = alpha * one_hot_target + (1 - alpha) * wrong_mass_weight * elem_weight
    sum_coeff = torch.matmul(softmax_vals.unsqueeze(dim=1), prox_dom[targets]).squeeze(dim=1)

    mass_weights = mass_weights.detach()
    sum_coeff = sum_coeff.detach()

    loss_val = -1 * torch.sum(mass_weights * torch.log(softmax_vals / sum_coeff)) / num_examples
    del softmax_vals, targets, sum_coeff, one_hot_target, one_hot_target_comp, wrong_mass_weight, \
        mass_weights, elem_weight
    return loss_val


def OCE_loss_final_with_prox_dynamic_L7_with_alpha_ablation_conf(softmax_vals, targets, prox_dom, norm_prox_mat,
                                                                 labels_num, alpha, device):
    num_examples = targets.shape[0]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    elem_weight = torch.tensor(norm_prox_mat[:, targets]).t()
    # wrong_mass_weight = torch.sum(one_hot_target_comp * softmax_vals, dim=1).unsqueeze(dim=1) * one_hot_target_comp
    mass_weights = alpha * one_hot_target + (1 - alpha) * elem_weight
    sum_coeff = torch.matmul(softmax_vals.unsqueeze(dim=1), prox_dom[targets]).squeeze(dim=1)

    mass_weights = mass_weights.detach()
    sum_coeff = sum_coeff.detach()

    loss_val = -1 * torch.sum(mass_weights * torch.log(softmax_vals / sum_coeff)) / num_examples
    del softmax_vals, targets, sum_coeff, one_hot_target, one_hot_target_comp, \
        mass_weights, elem_weight
    return loss_val


def OCE_loss_final_with_prox_dynamic_L7_with_alpha_ablation_prox(softmax_vals, targets, prox_dom, norm_prox_mat,
                                                                 labels_num, alpha, device):
    num_examples = targets.shape[0]
    one_hot_target = torch.tensor(np.zeros([num_examples, labels_num])).to(device)
    one_hot_target[range(num_examples), targets] = 1
    one_hot_target_comp = 1 - one_hot_target
    # elem_weight = torch.tensor(norm_prox_mat[:, targets]).t()
    wrong_mass_weight = torch.sum(one_hot_target_comp * softmax_vals, dim=1).unsqueeze(dim=1) * one_hot_target_comp
    mass_weights = alpha * one_hot_target + (1 - alpha) * wrong_mass_weight
    sum_coeff = torch.matmul(softmax_vals.unsqueeze(dim=1), prox_dom[targets]).squeeze(dim=1)

    mass_weights = mass_weights.detach()
    sum_coeff = sum_coeff.detach()

    loss_val = -1 * torch.sum(mass_weights * torch.log(softmax_vals / sum_coeff)) / num_examples
    del softmax_vals, targets, sum_coeff, one_hot_target, one_hot_target_comp, wrong_mass_weight, \
        mass_weights
    return loss_val

#################### UNTIL HERE ####################


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config, pos_tag_embedding = False, senti_embedding = False, polarity_embedding = False):
        super(RobertaEmbeddings, self).__init__(config, pos_tag_embedding=pos_tag_embedding, senti_embedding=senti_embedding, polarity_embedding=polarity_embedding)
        self.padding_idx = 1

    def forward(self, input_ids, token_type_ids=None, position_ids=None, pos_tag_ids=None, senti_word_ids=None, polarity_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = torch.arange(self.padding_idx+1, seq_length+self.padding_idx+1, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return super(RobertaEmbeddings, self).forward(input_ids,
                                                      token_type_ids=token_type_ids,
                                                      position_ids=position_ids,
                                                      pos_tag_ids=pos_tag_ids,
                                                      senti_word_ids=senti_word_ids,
                                                      polarity_ids=polarity_ids)


class RobertaModel(BertModel):

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, pos_tag_embedding=False, senti_embedding=False, polarity_embedding=False):
        super(RobertaModel, self).__init__(config, pos_tag_embedding=pos_tag_embedding, senti_embedding=senti_embedding, polarity_embedding=polarity_embedding)

        self.embeddings = RobertaEmbeddings(config, pos_tag_embedding=pos_tag_embedding, senti_embedding=senti_embedding, polarity_embedding=polarity_embedding)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, pos_ids=None, senti_word_ids=None, polarity_ids=None,):
        if input_ids[:, 0].sum().item() != 0:
            logger.warning("A sequence with no special tokens has been passed to the RoBERTa model. "
                           "This model requires special tokens in order to work. "
                           "Please specify add_special_tokens=True in your encoding.")
        return super(RobertaModel, self).forward(input_ids,
                                                 attention_mask=attention_mask,
                                                 token_type_ids=token_type_ids,
                                                 position_ids=position_ids,
                                                 head_mask=head_mask,
                                                 pos_ids=pos_ids,
                                                 senti_word_ids=senti_word_ids,
                                                 polarity_ids=polarity_ids)



class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super(RobertaLMHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias

        return x


class RobertaForSequenceClassification(BertPreTrainedModel):

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, pos_tag_embedding=False, senti_embedding=False, polarity_embedding=False,
                 loss_type='CrossEntropy', alpha=0.6):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, pos_tag_embedding=pos_tag_embedding,
                                    senti_embedding=senti_embedding, polarity_embedding=polarity_embedding)
        self.classifier = RobertaClassificationHead(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss_type = loss_type

        dist_dict = {4: 2322, 2: 2218, 3: 1624, 5: 1288, 1: 1092}  # TODO: Consider change this
        self.alpha = alpha
        self.inv_prox_mat = torch.tensor(create_prox_mat(dist_dict, True)).to(self.device) \
            if dist_dict is not None else None
        self.prox_mat = torch.tensor(create_prox_mat(dist_dict, False)).to(self.device) \
            if dist_dict is not None else None
        self.mass_encoder = create_mass_encoder(self.num_labels).to(self.device)
        self.prox_dom = create_prox_dom(self.prox_mat).to(self.device)
        self.norm_prox_mat = f.normalize(self.prox_mat, p=1, dim=0)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, pos_tag_ids, senti_word_ids, polarity_ids, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               pos_ids=pos_tag_ids,
                               senti_word_ids=senti_word_ids,
                               polarity_ids=polarity_ids
                               )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.loss_type == 'CrossEntropy':
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.loss_type == 'OCE_loss_final_with_prox_dynamic_L7_with_alpha':
                    logits_shaped = logits.view(-1, self.num_labels)
                    softmax_values = Func.softmax(logits_shaped, dim=1)
                    loss = OCE_loss_final_with_prox_dynamic_L7_with_alpha(softmax_values, labels.view(-1),
                                                                          self.prox_dom, self.norm_prox_mat,
                                                                          self.num_labels, self.alpha,
                                                                          self.device)
                elif self.loss_type == 'OCE_loss_final_with_prox_dynamic_L7_with_alpha_ablation_conf':
                    logits_shaped = logits.view(-1, self.num_labels)
                    softmax_values = Func.softmax(logits_shaped, dim=1)
                    loss = OCE_loss_final_with_prox_dynamic_L7_with_alpha_ablation_conf(softmax_values, labels.view(-1),
                                                                                        self.prox_dom, self.norm_prox_mat,
                                                                                        self.num_labels, self.alpha,
                                                                                        self.device)

                elif self.loss_type == 'OCE_loss_final_with_prox_dynamic_L7_with_alpha_ablation_prox':
                    logits_shaped = logits.view(-1, self.num_labels)
                    softmax_values = Func.softmax(logits_shaped, dim=1)
                    loss = OCE_loss_final_with_prox_dynamic_L7_with_alpha_ablation_prox(softmax_values, labels.view(-1),
                                                                                        self.prox_dom, self.norm_prox_mat,
                                                                                        self.num_labels, self.alpha,
                                                                                        self.device)

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaForMultiLabelClassification(BertPreTrainedModel):

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, pos_tag_embedding=False, senti_embedding=False, polarity_embedding=False):
        super(RobertaForMultiLabelClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, pos_tag_embedding=pos_tag_embedding,
                                      senti_embedding=senti_embedding,
                                      polarity_embedding=polarity_embedding)
        self.classifier = RobertaClassificationHead(config)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, input_ids, pos_tag_ids, senti_word_ids, polarity_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               pos_ids=pos_tag_ids,
                               senti_word_ids=senti_word_ids,
                               polarity_ids=polarity_ids
                               )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        logits = self.sigmoid_layer(logits)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = BCELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaForMultipleChoice(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, pos_tag_embedding=False, senti_embedding=False, polarity_embedding=False):
        super(RobertaForMultipleChoice, self).__init__(config)

        self.roberta = RobertaModel(config, pos_tag_embedding=pos_tag_embedding,
                                      senti_embedding=senti_embedding,
                                      polarity_embedding=polarity_embedding)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids, pos_tag_ids, senti_word_ids, polarity_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
    ):

        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_pos_tag_ids = pos_tag_ids.view(-1, pos_tag_ids.size(-1))
        flat_senti_word_ids = senti_word_ids.view(-1, senti_word_ids.size(-1))
        flat_polarity_ids = polarity_ids.view(-1, polarity_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            pos_ids=flat_pos_tag_ids,
            senti_word_ids=flat_senti_word_ids,
            polarity_ids=flat_polarity_ids
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForTokenClassification(BertPreTrainedModel):

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, pos_tag_embedding=False, senti_embedding=False, polarity_embedding=False):
        super(RobertaForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, pos_tag_embedding=pos_tag_embedding,
                                      senti_embedding=senti_embedding,
                                      polarity_embedding=polarity_embedding)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        pos_tag_ids=None,
        senti_word_ids=None,
        polarity_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            pos_ids=pos_tag_ids,
            senti_word_ids=senti_word_ids,
            polarity_ids=polarity_ids
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)