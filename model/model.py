import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class Model(nn.Module):
    def __init__(self, dropout=0.1, num_classes=53, name='model'):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.name = name

        self.dropout = nn.Dropout(dropout)

        self.BERT = BertModel.from_pretrained("bert-base-uncased")
        self.EntityBert = BertModel.from_pretrained("bert-base-uncased")

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.BERT.to(self.device)
        self.EntityBert.to(self.device)

        self.Decoder_BERT = nn.Linear(768, num_classes)
        self.entity_decoder_bert = nn.Linear(768, num_classes)
        self.final_label_classi = FCLayer(768 * 2, num_classes, dropout, use_activation=False)


    def decode_new(self, x, attentionmask, tokentype, total_shape, entity_batch):

        bert = self.BERT(x, attentionmask, tokentype)[1]
        batch_sent_emb = self.dropout(bert)
        batch_sent_emb = torch.tanh(batch_sent_emb)
        batch_e = self.Decoder_BERT(batch_sent_emb)
        batch_score = []
        output = self.EntityBert(entity_batch)[1]
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            o = self.Decoder_BERT(self.dropout(bag_rep))
            o = F.softmax(o, 1)
            o = F.softmax(o.diag(), -1)
            index = torch.argmax(o)
            f_rep = torch.cat((bag_rep[index], output[i]))
            f = self.final_label_classi(f_rep)

            s = self.entity_decoder_bert(self.dropout(output[i]))
            s = F.softmax(s, -1)
            pred = 0.6 * f + 0.4 * s
            pred = F.softmax(pred, -1)
            batch_score.append(pred)
        batch_p = torch.stack(batch_score)
        return batch_p

    def Train_cat(self, x, attentionmask, tokentype, total_shape, y_batch, entity_batch):

        outputs = self.BERT(x, attentionmask, tokentype)
        bert = outputs[1]
        batch_sent_emb = self.dropout(bert)
        batch_sent_emb = torch.tanh(batch_sent_emb)
        batch_e = self.Decoder_BERT(batch_sent_emb)
        output = self.EntityBert(entity_batch)[1]
        batch_score = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            o = self.Decoder_BERT(self.dropout(bag_rep))

            index = torch.argmax(o.diag())
            f_rep = torch.cat((bag_rep[index], output[i]))
            f = self.final_label_classi(f_rep)
            batch_score.append(f)
        res_bag = torch.stack(batch_score)

        res_entity = self.entity_decoder_bert(self.dropout(output))
        loss2 = nn.functional.cross_entropy(res_entity, y_batch)

        loss1 = nn.functional.cross_entropy(res_bag, y_batch)

        return loss1 + loss2

    def forward(self, x, attentionmask, tokentype, total_shape, y_batch, entity_batch):

        bert = self.BERT(x, attentionmask, tokentype)[1]
        batch_sent_emb = self.dropout(bert)
        batch_sent_emb = torch.tanh(batch_sent_emb)
        batch_e = self.Decoder_BERT(batch_sent_emb)
        batch_score = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            o = self.Decoder_BERT(self.dropout(bag_rep))
            batch_score.append(o.diag())
        res_bag = torch.stack(batch_score)
        loss1 = nn.functional.cross_entropy(res_bag, y_batch)

        output = self.EntityBert(entity_batch)[1]
        res_entity = self.entity_decoder_bert(self.dropout(output))
        loss2 = nn.functional.cross_entropy(res_entity, y_batch)

        return loss1 + loss2
