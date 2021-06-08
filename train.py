import pickle
import math
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
from preprocess.preprocess import DocumentContainer
from model.model import Model
from transformers import AdamW


def bags_decompose(data_bags):
    bag_sent = [data_bag.sentences for data_bag in data_bags]
    bag_label = [data_bag.label for data_bag in data_bags]
    bag_len = [data_bag.length for data_bag in data_bags]
    bag_entity = [data_bag.bag_token for data_bag in data_bags]
    return [bag_label, bag_sent, bag_len, bag_entity]


def eval(model, testset, args):
    [test_label, test_sents, test_len, test_entity] = bags_decompose(testset)

    print('testing...')
    y_true = []
    y_scores = []
    batch = args.batch_size_eval

    index = 0
    addlength = 0
    temp_input = []
    while index < len(test_label):
        length = test_len[index]
        if addlength + length <= batch:
            temp_input.append(index)
            index = index + 1
            addlength = addlength + length
            continue

        total_shape = []
        total_num = 0
        total_inputId = []
        total_attentionMask = []
        total_tokenType = []
        total_entity = []
        total_y = []

        for k in temp_input:
            total_shape.append(total_num)
            total_num += len(test_sents[k])
            total_entity.append(test_entity[k])

            temp = [0] * model.num_classes
            temp[test_label[k]] = 1
            total_y.append(temp)

            for l in range(len(test_sents[k])):
                total_inputId.append(test_sents[k][l]['input_ids'])
                total_attentionMask.append(test_sents[k][l]['attention_mask'])
                total_tokenType.append(test_sents[k][l]['token_type_ids'])
                # total_pos.append(test_sents[k][l]['entity_pos'])

        total_shape.append(total_num)

        total_inputId = np.array(total_inputId)
        total_attentionMask = np.array(total_attentionMask)
        total_tokenType = np.array(total_tokenType)
        total_entity = np.array(total_entity)
        total_y = np.array(total_y)

        total_inputId = Variable(torch.from_numpy(total_inputId)).cuda()
        total_attentionMask = Variable(torch.from_numpy(total_attentionMask)).cuda()
        total_tokenType = Variable(torch.from_numpy(total_tokenType)).cuda()
        entity_batch = Variable(torch.from_numpy(total_entity)).cuda()

        with torch.no_grad():
            batch_p = model.decode_new(total_inputId, total_attentionMask, total_tokenType, total_shape,
                                             entity_batch)
            batch_p = batch_p.cpu().data.numpy()

        y_true.append(total_y[:, 1:])
        y_scores.append(batch_p[:, 1:])

        addlength = length
        temp_input = [index]
        index = index + 1

    y_true = np.concatenate(y_true).reshape(-1)
    y_scores = np.concatenate(y_scores).reshape(-1)

    return y_true, y_scores


def AUC_and_PN(model, datasets, args):
    model.eval()

    testdata = datasets

    y_true, y_scores = eval(model, testdata, args)

    np.save('result/' + args.modelname + '_true.npy', y_true)
    np.save('result/' + args.modelname + '_scores.npy', y_scores)

    AUC_all = average_precision_score(y_true, y_scores)

    print('AUC value:', AUC_all)


def trainModel(model, train_data, args, dataset):
    model.train()
    [train_label, train_sents, train_len, train_entity] = bags_decompose(train_data)
    lr = args.init_lr
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.005,
            # 对于除去bias和norm的参数进行L2正则化
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8,
    )

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print("Training:", str(now))
    temp_order = list(range(len(train_label)))
    batch = args.batch_size_train
    for epoch in range(args.train_epoch):
        lr = lr * (args.train_epoch - epoch) / args.train_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        np.random.shuffle(temp_order)

        index = 0
        addlength = 0
        temp_input = []
        while index < len(temp_order):
            length = train_len[temp_order[index]]
            if addlength + length <= batch:
                temp_input.append(temp_order[index])
                index = index + 1
                addlength = addlength + length
                continue

            total_shape = []
            total_num = 0
            total_inputId = []
            total_attentionMask = []
            total_tokenType = []
            total_y = []
            total_entity = []

            for k in temp_input:
                total_shape.append(total_num)
                total_num += len(train_sents[k])
                total_y.append(train_label[k])
                total_entity.append(train_entity[k])
                for j in range(len(train_sents[k])):
                    total_inputId.append(train_sents[k][j]['input_ids'])
                    total_attentionMask.append(train_sents[k][j]['attention_mask'])
                    total_tokenType.append(train_sents[k][j]['token_type_ids'])

            total_shape.append(total_num)

            total_inputId = np.array(total_inputId)
            total_attentionMask = np.array(total_attentionMask)
            total_tokenType = np.array(total_tokenType)
            total_y = np.array(total_y)
            total_entity = np.array(total_entity)

            total_inputId = Variable(torch.from_numpy(total_inputId)).cuda()
            total_attentionMask = Variable(torch.from_numpy(total_attentionMask)).cuda()
            total_tokenType = Variable(torch.from_numpy(total_tokenType)).cuda()
            y_batch = Variable(torch.from_numpy(total_y)).cuda()
            entity_batch = Variable(torch.from_numpy(total_entity)).cuda()

            loss = model.Train_cat(total_inputId, total_attentionMask, total_tokenType, total_shape, y_batch,
                                  entity_batch)

            model.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            addlength = length
            temp_input = [temp_order[index]]
            index = index + 1

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print(str(now), epoch)
        AUC_and_PN(model, dataset, args)
        model.train()

    torch.save({'model': model.state_dict()}, 'result/' + model.name + '.model')

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train code')
    parser.add_argument('--train_file', default='preprocess/20/train.pkl', help='path to training file')
    parser.add_argument('--test_file', default='preprocess/20/test.pkl', help='path to test file')
    parser.add_argument('--batch_size_train', type=int, default=20, help='batch size for training')
    parser.add_argument('--batch_size_eval', type=int, default=200, help='batch size for eval')
    parser.add_argument('--init_lr', type=float, default=1e-6, help='initial learning rate')
    parser.add_argument('--train_epoch', type=int, default=5, help='epoch for training')
    parser.add_argument('--num_classes', type=int, default=53, help='class number for the dataset')
    parser.add_argument('--modelname', type=str, default="BL&GE_BERT", help='model name')
    parser.add_argument('--modelpath', type=str, default="result/pretrain_BERT.model", help='path to pretrain model file')

    args = parser.parse_args()
    print(args)

    train_data = pickle.load(open(args.train_file, 'rb'), encoding='utf-8')
    testdata = pickle.load(open(args.test_file, 'rb'), encoding='utf-8')

    print('modelname: ', args.modelname)

    model = Model(num_classes=args.num_classes, name=args.modelname)
    model.cuda()

    config = torch.load(args.modelpath)
    model.load_state_dict(config['model'], strict=False)

    trainModel(model=model, train_data=train_data, args=args, dataset=testdata)
