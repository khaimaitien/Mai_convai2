import os, datetime
import data_reader
import torch
import numpy as np
import convai_model
import cand_retrieval
import utility


def load_model(use_cuda, model_folder):
    model_path = os.path.join(model_folder, 'model.dat')
    print ('load model from %s' % model_path)
    setting = data_reader.read_setting(os.path.join(model_folder, 'setting.json'))
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


class WrapConverse(object):
    def __init__(self, use_cuda=False):
        cur_folder = os.path.dirname(os.path.abspath(__file__))
        model_folder = os.path.join(cur_folder, 'model')
        self.use_cuda = use_cuda
        self.model = load_model(self.use_cuda, model_folder)
        self.vocab_dic = data_reader.load_vocab(data_reader.get_vocab_path())

    def get_ranked_candidates(self, converse):
        temp_converse = utility.convert_converse_to_wid(self.vocab_dic, converse)
        scores = utility.get_prediction_from_converse(temp_converse, self.use_cuda, self.model)
        ### rank by scores ####
        score_list = scores.data.tolist()
        score_np = np.array(score_list) # N * k (N = number of questions, k = number of candidates)
        score_indexs = np.argsort(score_np)
        result = []
        N = len(score_list)
        for i in range(N):
            items = list(score_indexs[i])
            items.reverse()
            result.append(items)
        #print ('total call: %d, leng = %d' % (self.count, len()))
        return result

    def get_response(self, profile, question):
        """

        :param profile: list of sentences: [['a', 'b'], ['c', 'd']]
        :param question: [['a', 'b'], ['c', 'd']]
        :return:
        """
        p_text_list = [' '.join(item) for item in profile]
        p_text = ' '.join(p_text_list)
        temp_cand = cand_retrieval.get_documents(p_text, 200)
        cands = [item.split(' ') for item in temp_cand]
        cand_sorts = self.get_ranked_candidates({'question': question, 'profile': profile, 'cand': [cands]})
        b_indices = cand_sorts[0]
        response = cands[b_indices[0]]
        return ' '.join(response)


def get_wrap_converse():
    cur_folder = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(cur_folder, 'model')
    setting_path = os.path.join(model_folder, 'setting.json')
    setting = data_reader.read_json(setting_path)
    use_cuda = setting['use_cuda']
    result = WrapConverse(use_cuda)
    return result

WRAP_CONVERSE = get_wrap_converse()


def get_response(profile, question):
    sens = profile
    if type(sens[0]) is not list:
        temp_profile = []
        for sen in sens:
            temp_profile.append(sen.split(' '))
        tok_profile = temp_profile
    else:
        tok_profile = sens
    tok_question = question
    if type(question) is not list:
        p_question = data_reader.preprocess_rm_period(question)
        tok_question = p_question.split(' ')
    return WRAP_CONVERSE.get_response(tok_profile, [tok_question])



def tokenize_conver(conver):
    result = {}
    question = conver['question']
    question = data_reader.preprocess_rm_period(question)
    result['question'] = [question.split(' ')]
    cand_items = conver['cand']
    cands = []
    rm_items = []
    for item in cand_items:
        temp_item = data_reader.preprocess_rm_period(item)
        temp_item = temp_item.split(' ')
        rm_items.append(temp_item)
    cands.append(rm_items)
    result['cand'] = cands
    sens = conver['profile']
    if type(sens[0]) is not list:
        temp_profile = []
        for sen in sens:
            temp_profile.append(sen.split(' '))
        result['profile'] = temp_profile
    else:
        result['profile'] = sens
    return result

def get_ranked_indices(conver):
    return WRAP_CONVERSE.get_ranked_candidates(conver)

def get_ranked_indices_for_one_question(conver):
    temp_conver = tokenize_conver(conver)
    indices = get_ranked_indices(temp_conver)
    return indices[0]


def test_ranked_model():
    converses, _ = data_reader.read_training_data('data/valid_self_original.txt', False)
    true_count = 0
    total_count = 0
    t1 = datetime.datetime.now()
    count = 0
    for converse in converses:
        indices = get_ranked_indices(converse) # N * k
        true_indexs = converse['indexs']# N
        for i in range(len(true_indexs)):
            if true_indexs[i] == indices[i][0]:
                true_count += 1
        total_count += len(true_indexs)
        count += 1
        if count % 10 == 1:
            print ('count = %d' % count)
    ratio = float(true_count)/total_count
    t2 = datetime.datetime.now()
    print ('time for evaluating model: %f seconds' % (t2 - t1).total_seconds())
    print ('accuracy %d/%d = %f' % (true_count, total_count, ratio))


def test_auto_gen_response():
    pro_sens = [
        'i read twenty books a year',
        'i\'m a stunt double as my second job',
        'i only eat kosher.',
        'i was raised in a single parent household'
    ]
    profile = [item.split(' ') for item in pro_sens]
    question = 'i just got done watching a horror movie'
    result = get_response(profile, question)
    print (result)
    while True:
        input_str = input('question: ')
        result = get_response(profile, input_str)
        print (result)


if __name__ == '__main__':
    #test_auto_gen_response()
    test_ranked_model()

