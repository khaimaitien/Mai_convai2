import numpy as np
import re, os, json
import convai_model
import torch
import utility
import sys
import datetime
import os
import data_reader

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODER_FOLDER = os.path.join(CURRENT_FOLDER, 'model2')
if not os.path.exists(MODER_FOLDER):
    os.mkdir(MODER_FOLDER)


def get_random_permute(N):
    per_path = '%d_permutation.out' % N
    if not os.path.exists(per_path):
        permute = np.random.permutation(N)
        indexs = [str(index) for index in permute]
        f = open(per_path, 'w')
        f.write(' '.join(indexs))
        f.close()
    f = open(per_path, 'r')
    text = f.read()
    f.close()
    tgs = text.strip().split(' ')
    indexs = [int(tg) for tg in tgs]
    return indexs


def get_random_permutation(items):
    N = len(items)
    permute = get_random_permute(N)
    result = [items[index] for index in permute]
    return result


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def get_model_path(epo, count):
    return os.path.join(MODER_FOLDER, 'model_%d_%d.dat' % (epo, count))


def get_final_model_path():
    return os.path.join(MODER_FOLDER, 'final_model.dat')

def get_best_model_path():
    return os.path.join(MODER_FOLDER, 'best_model.dat')

def get_test_report_path():
    return os.path.join(MODER_FOLDER, 'report.txt')


def append_to_file(file_path, content):
    if not os.path.exists(file_path):
        f = open(file_path, 'w')
        f.write(content + '\n')
        f.close()
    else:
        f = open(file_path, 'a')
        f.write('%s\n' % content)
        f.close()


def save_and_eval_test(model, test_convers, use_cuda, epo, count):
    save_model(model, get_model_path(epo, count))
    test_total_sens, test_total_true = eval_model_with_data(use_cuda, model, test_convers)
    true_ratio = float(test_total_true) / test_total_sens
    report_line = 'epo: %d, b_count: %d, test_convers: %d, total_sens: %d, total_true: %s, ratio = %f' % (
        epo, count, len(test_convers), test_total_sens, test_total_true, true_ratio
    )
    print (report_line)
    append_to_file(get_test_report_path(), report_line)


def remove_cands_in_converse(conver, K):
    cands = conver['cand']
    new_cands = []
    for cand in cands:
        new_cands.append(cand[: K])
    conver['cand'] = new_cands


def train_model(setting):
    #save_setting(setting)
    convers, _ = data_reader.read_training_data('data/train_self_original.txt', True, 20)
    vocab_dic = data_reader.load_vocab(data_reader.get_vocab_path())
    w2vec_init = data_reader.load_w2vec_matrix()
    convers = get_random_permutation(convers)
    print ('number of conversations: %d' % len(convers))
    print ('size of w2vec matrix: ', w2vec_init.shape)
    setting['vocab_size'] = w2vec_init.shape[0]
    setting['word_dim'] = w2vec_init.shape[1]
    data_reader.save_setting(setting, os.path.join(MODER_FOLDER, 'setting.json'))
    setting['embed_init'] = w2vec_init
    lamda = setting['lamda']
    learning_rate = setting['learning_rate']
    model = convai_model.RankingModel(setting)
    use_cuda = setting['use_cuda']
    if use_cuda:
        model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    epo_num = setting['epo_num']
    tr_1 = datetime.datetime.now()
    ### remove dev --> only 20 #####
    dev_convers = convers[: 100]
    for d_con in dev_convers:
        remove_cands_in_converse(d_con, 20)
    print ('average of number of candidate dev: %d' % len(dev_convers[0]['cand'][0]))
    tr_convers = convers[100: ]
    print ('average of number of candidate train: %d' % len(tr_convers[0]['cand'][0]))
    tr_convers = [utility.convert_converse_to_wid(vocab_dic, conver) for conver in tr_convers]
    dev_convers = [utility.convert_converse_to_wid(vocab_dic, conver) for conver in dev_convers]
    #### check accuracy of test here ###
    test_convers, _ = data_reader.read_training_data('data/valid_self_original.txt', True)
    test_convers = [utility.convert_converse_to_wid(vocab_dic, conver) for conver in test_convers]
    append_to_file(get_test_report_path(), 'train: %d conversations, dev: %d, test: %d' % (len(tr_convers), len(dev_convers), len(test_convers)))
    print ('training convers: %d' % len(tr_convers))
    b_count = 0
    for epo in range(epo_num):
        print ('at epoch: %d' % epo)
        count = 0
        inter_d1 = datetime.datetime.now()
        for conver in tr_convers:
            ### find loss from scores and true_indexs###
            optimizer.zero_grad()
            loss, _ = utility.get_loss_from_converse(conver, use_cuda, model, lamda)
            loss.backward()
            optimizer.step()
            #print ('time for 1 conversation = %f' % (b2 - b1).total_seconds())
            if count % 5000 == 1:
                print ('compute loss at dev at %d of epo %d' % (count, epo))
                d_t1 = datetime.datetime.now()
                total_loss = 0
                total_true_pre = 0
                total_sens = 0
                for d_conver in dev_convers:
                    temp_loss, temp_true_count = utility.get_loss_from_converse(d_conver, use_cuda, model, lamda, True)
                    total_loss += temp_loss
                    total_true_pre += temp_true_count
                    total_sens += len(d_conver['question_leng'])
                dev_loss = total_loss.data[0]
                d_t2 = datetime.datetime.now()
                delta_d = (d_t2 - d_t1).total_seconds()
                accuracy = float(total_true_pre) /float(total_sens)
                print ('loss of dev at %d of epo %d is %f, accuracy %d/%d = %f, time = %f' % (count, epo, dev_loss, total_true_pre, total_sens, accuracy, delta_d))
                inter_d2 = datetime.datetime.now()
                print ('tim for traingin 5000 convers: %f seconds' % (inter_d2 - inter_d1).total_seconds())
                inter_d1 = datetime.datetime.now()
            b_count += 1
            count += 1
            if b_count % 15000 == 1:
                save_and_eval_test(model, test_convers, use_cuda, epo, b_count)

    tr_2 = datetime.datetime.now()
    delta_time = (tr_2 - tr_1).total_seconds()
    print ('time for training %d epo: %f seconds' % (epo_num, delta_time))
    save_model(model, get_final_model_path())
    print ('start eval newest model ...')
    eval_on_valid(True)
    print ('finished ...')


def get_final_model(use_cuda):
    model_path = get_best_model_path()
    print ('load best model from %s' % model_path)
    setting = data_reader.read_setting(os.path.join(MODER_FOLDER, 'setting.json'))
    w2vec_init = data_reader.load_w2vec_matrix()
    setting['vocab_size'] = w2vec_init.shape[0]
    setting['embed_init'] = w2vec_init
    setting['word_dim'] = w2vec_init.shape[1]
    model = convai_model.RankingModel(setting)
    if use_cuda:
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        saved_state = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state)
    return model


def eval_model_with_data(use_cuda, model, convers):
    total_sens = 0
    total_true = 0
    count = 0
    print ('start evaluating %d conversations' % len(convers))
    for j in range(len(convers)):
        conver = convers[j]
        # print ('for converse: ', conver)
        just_forward = False
        if use_cuda:
            just_forward = True
        scores = utility.get_prediction_from_converse(conver, use_cuda, model, just_forward)
        indexs = conver['index']
        true_pre = 0
        _, max_indexs = torch.max(scores, 1)  # N
        max_indexs = max_indexs.data.tolist()
        for i in range(len(indexs)):
            if max_indexs[i] == indexs[i]:
                # print ('ok at conver: %d, record: %d' % (j, i))
                true_pre += 1
        total_true += true_pre
        total_sens += len(conver['question_leng'])
        if count == 1000:
            print ('count = %d' % count)
    return total_sens, total_true



def eval_on_valid(use_cuda):
    o_convers, _ = data_reader.read_training_data('data/valid_self_original.txt', False)
    vocab_dic = data_reader.load_vocab(data_reader.get_vocab_path())
    model = get_final_model(use_cuda)
    convers = [utility.convert_converse_to_wid(vocab_dic, conver) for conver in o_convers]
    total_sens, total_true = eval_model_with_data(use_cuda, model, convers)
    acc_ratio = float(total_true)/float(total_sens)
    print ('final accuracy: %d/%d = %f' % (total_sens, total_true, acc_ratio))


def run_train1(use_cuda):
    setting = {
        'sim_dim': 512,
        'retrain_emb': False,
        'use_cuda': use_cuda,
        'alpha': 0.5,
        'learning_rate': 0.001,
        'epo_num': 15,
        'build_dic': False,
        'lamda': 0.3,
        'sim_dim': 1024,
        'att_type': 'concat',
        'profile_dim': 512,
        'att_dim': 512
    }
    train_model(setting)


def main():
    if len(sys.argv) != 3:
        print ('usage: python k_train.py train/eval use_cuda')
        sys.exit(1)
    mode = sys.argv[1]
    use_cuda = False
    if sys.argv[2].strip().lower() == 'true':
        use_cuda = True
    if mode == 'eval':
        eval_on_valid(use_cuda)
    elif mode == 'train':
        run_train1(use_cuda)

if __name__ == '__main__':
    main()