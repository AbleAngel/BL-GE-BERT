import argparse
import codecs
import pickle

from transformers import BertTokenizer


class DocumentContainer(object):
    def __init__(self, bag_token, sentences, label, length):
        self.bag_token = bag_token
        self.sentences = sentences
        self.label = label
        self.length = length


def relation2id(f='../NYT_data/relation2id.txt'):
    rel2id = codecs.open(f, 'r', 'utf-8')
    table = {}
    while True:
        line = rel2id.readline().strip()
        if not line:
            break
        line = line.split()
        table[line[0]] = int(line[1])
    return table


def bulid_token_in_sent(dic, max_bag_len, max_seq_len=512, f='../NYT_data/train.txt'):
    all_data = []
    sent_file = codecs.open(f, 'r', 'utf-8')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bag_entitypair = None
    sentences = []
    bag_label = None
    while True:
        line = sent_file.readline().strip()
        if not line:
            break
        line = line.split()
        entity1 = line[2]
        entity2 = line[3]
        entity_pair = entity1 + '#' + entity2
        label = dic.get(line[4], 0)
        sentence = " ".join(line[5:])
        sentence = sentence.replace(entity1, '$' + entity1 + '$')
        sentence = sentence.replace(entity2, '#' + entity2 + '#')
        sentence = sentence.replace('###END###', '')
        token = tokenizer(sentence)
        # pos = [0, 0, 0, 0]
        # add pad token and cut long sentence
        if len(token['input_ids']) <= max_seq_len:
            pad_len = max_seq_len - len(token['input_ids'])
            token['input_ids'] = token['input_ids'] + ([0] * pad_len)
            token['attention_mask'] = token['attention_mask'] + ([0] * pad_len)
            token['token_type_ids'] = token['token_type_ids'] + ([0] * pad_len)
        else:
            token['input_ids'] = token['input_ids'][0:max_seq_len - 1] + [102]
            # 102 is id of SEP token. Here I need to add one SEP after part sentences
            token['attention_mask'] = token['attention_mask'][0:max_seq_len]
            token['token_type_ids'] = token['token_type_ids'][0:max_seq_len]
        # sent_token, token, label

        # if token['input_ids'].count(1001) == 2:
        #     pos[0] = token['input_ids'].index(1001)
        #     pos[1] = token['input_ids'].index(1001, pos[0]+1)
        # if token['input_ids'].count(1002) == 2:
        #     pos[2] = token['input_ids'].index(1002)
        #     pos[3] = token['input_ids'].index(1002, pos[2]+1)
        # token['entity_pos'] = pos

        assert len(token['input_ids']) == max_seq_len, "Error id"
        assert len(token['attention_mask']) == max_seq_len, "Error attention mask"
        assert len(token['token_type_ids']) == max_seq_len, "Error token type"

        if bag_entitypair is None:
            bag_entitypair = entity_pair
            sentences.append(token)
            bag_label = label
        elif entity_pair == bag_entitypair and len(sentences) < max_bag_len:
            sentences.append(token)
        else:
            entity_token = tokenizer(bag_entitypair)
            pad_len = 100 - len(entity_token['input_ids'])
            entity_token['input_ids'] = entity_token['input_ids'] + ([0] * pad_len)

            assert len(entity_token['input_ids']) == 100, "Error id"

            new_ins = DocumentContainer(bag_token=entity_token['input_ids'], sentences=sentences, label=bag_label,
                                        length=len(sentences))
            all_data.append(new_ins)
            bag_entitypair = entity_pair
            sentences = [token]
            bag_label = label

    entity_token = tokenizer(bag_entitypair)
    pad_len = 100 - len(entity_token['input_ids'])
    entity_token['input_ids'] = entity_token['input_ids'] + ([0] * pad_len)

    assert len(entity_token['input_ids']) == 100, "Error id"

    new_ins = DocumentContainer(bag_token=entity_token['input_ids'], sentences=sentences, label=bag_label,
                                length=len(sentences))
    all_data.append(new_ins)

    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess code')
    parser.add_argument('--bag_max_length', type=int, default=20, help='one bag max length')

    args = parser.parse_args()

    rel_dictinary = relation2id('../NYT_data/relation2id.txt')
    length = args.bag_max_length

    train_data = bulid_token_in_sent(rel_dictinary, length, f='../NYT_data/train.txt')
    f = open('./' + str(args.bag_max_length) + '/train.pkl', 'wb')
    pickle.dump(train_data, f, -1)
    f.close()

    test_data = bulid_token_in_sent(rel_dictinary, length, f='../NYT_data/test.txt')
    f = open('./' + str(args.bag_max_length) + '/test.pkl', 'wb')
    pickle.dump(test_data, f, -1)
    f.close()
