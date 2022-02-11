import torch
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, DistilBertTokenizer
from keras.preprocessing.sequence import pad_sequences


class TextDataReader:
    def __init__(self, df, embeddings_version, max_len, text_col,
                 label_col, key_col, sub_nn=None, nn_type='primary'):
        self.texts = list(df[text_col].values)
        self.df = df
        self.labels = self.adjust_labels(label_col)
        self.key_ids = torch.tensor(df[key_col].values)
        self.tokenizer = self.get_tokenizer(embeddings_version)
        self.inputs_ids = self.create_inputs_ids(max_len)
        self.masks = self.create_masks()
        self.nn_type = nn_type
        self.sub_nn = sub_nn
        self.labels_dist = self.create_labels_dist()
        self.flags = self.create_flags()

    @staticmethod
    def get_tokenizer(embeddings_version):
        if embeddings_version.startswith("bert-"):
            return BertTokenizer.from_pretrained(embeddings_version)
        elif embeddings_version.startswith("roberta-"):
            return RobertaTokenizer.from_pretrained(embeddings_version)
        elif embeddings_version.startswith("albert-"):
            return AlbertTokenizer.from_pretrained(embeddings_version)
        elif embeddings_version.startswith("distilbert-"):
            return DistilBertTokenizer.from_pretrained(embeddings_version)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.sub_nn != 'with_flags':
            return self.inputs_ids[index], self.masks[index], \
                   self.labels[index], self.key_ids[index],
        else:
            return self.inputs_ids[index], self.masks[index], \
                   self.labels[index], self.key_ids[index], \
                   self.labels_dist[index], self.flags[index]

    def adjust_labels(self, label_col):
        labels = self.df[label_col].values
        sorted_labels = list(set(labels))
        min_val, second_val = sorted_labels[0], sorted_labels[1]
        if second_val - min_val > 1:
            labels = list(map(lambda x: x - min_val if x == min_val else x - second_val + 1, labels))
        else:
            labels = list(map(lambda x: x - min_val, labels))
        return torch.tensor(labels)

    def create_inputs_ids(self, max_len):
        input_ids = []
        for text_input in self.texts:
            encoded_note = self.tokenizer.encode(text_input, add_special_tokens=True,
                                                 max_length=max_len, truncation=True)
            input_ids.append(encoded_note)
        input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long",
                                  value=0, truncating="post", padding="post")
        return torch.tensor(input_ids)

    def create_masks(self):
        masks = []
        for encoded_note in self.inputs_ids:
            mask = [int(token_idx > 0) for token_idx in encoded_note]
            masks.append(mask)
        return torch.tensor(masks)

    def create_labels_dist(self):
        if self.sub_nn != 'with_flags':
            return None
        else:
            return torch.stack([torch.tensor(dist) for dist in self.df['labels_dist'].values])

    def create_flags(self):
        if self.sub_nn != 'with_flags':
            return None
        else:
            return torch.tensor([torch.tensor(dist) for dist in self.df['flag']])
