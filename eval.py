import pickle
import argparse

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
from preprocess.preprocess import DocumentContainer
from model.model import Model


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
            # if np.isnan(batch_p).any():
            #     print(batch_p)
            #     input()

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

    np.save('result/' + args.outputname + '_true.npy', y_true)
    np.save('result/' + args.outputname + '_scores.npy', y_scores)

    AUC_all = average_precision_score(y_true, y_scores)

    print('AUC value:', AUC_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='made from ding angran')
    parser.add_argument('--test_file', default='preprocess/20/test.pkl', help='path to test file')
    parser.add_argument('--batch_size_eval', type=int, default=200, help='batch size for eval')
    parser.add_argument('--num_classes', type=int, default=53, help='class number for the dataset')
    parser.add_argument('--modelpath', type=str, default="result/BL&GE_BERT", help='model name')
    parser.add_argument('--outputname', type=str, default="BL&GE_BERT", help='model name')

    args = parser.parse_args()
    print(args)

    testdata = pickle.load(open(args.test_file, 'rb'), encoding='utf-8')

    datasets = testdata

    print('outputname: ', args.outputname)

    model = Model(num_classes=args.num_classes)
    model.cuda()

    config = torch.load(args.modelpath)
    model.load_state_dict(config['model'], strict=False)

    AUC_and_PN(model, datasets, args)
