import re, sys
import os, json
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import random
UNKNOWN_TOKEN = '<unnown>'
PADDING_TOKEN = '<paddingword>'

def extract_text_from_line_numb(line):
    match = re.search('\d+ your persona:', line)
    if match is None:
        match = re.search('\d+', line)
    if match is None:
        match = re.search('\d+ partner\'s persona:', line)
    if match is not None:
        end = match.end()
        rm_text = line[end: ].strip()
        return rm_text
    else:
        print ('cannot detect line number: %s' % line)
    return line


def is_partner_persona(text):
    match = re.search("\d+ partner's persona:", text)
    if match is not None:
        return True
    return False


def preprocess_rm_period(text):
    if text.endswith('.'):
        text = text[: -1].strip()
        return text
    return text


def update_dic(dic, item):
    if item not in dic:
        dic[item] = len(dic)


def get_item_ids(dic, items):
    for item in items:
        if item not in dic:
            dic[item] = len(dic)
    return [dic[item] for item in items]


def add_count_items(dic, items):
    for item in items:
        if item not in dic:
            dic[item] = 0
        dic[item] +=1

def find_true_index(answer, cands):
    for i in range(len(cands)):
        if cands[i] == answer:
            return i
    return -1


def get_candidate_set(convers):
    cand_set = set()
    for conver in convers:
        cand = conver['cand']
        for candlist in cand:
            for item in candlist:
                cand_set.add(item)
    return cand_set


def get_k_random_indices(N, k):
    result = set()
    while len(result) < k:
        index = random.randint(0, N - 1)
        result.add(index)
    return result


def gen_more_example(conver, poolist, K):
    questions = conver['question']
    answers = conver['answer']
    partners = conver['partner']
    if len(questions) > 1:
        new_questions = answers[: -1]
        new_answers = questions[1:]
        new_cands = []
        new_indexs = []
        for i in range(len(new_answers)):
            true_answer = new_answers[i]
            r_indices = get_k_random_indices(len(poolist), K)
            neg_examples = [poolist[index] for index in r_indices]
            neg_examples.append(true_answer)
            new_cands.append(neg_examples)
            new_indexs.append(len(neg_examples) - 1)
        new_conver = {'profile': partners, 'question': new_questions, 'answer': new_answers,
                      'cand': new_cands, 'indexs': new_indexs}
        return new_conver
    return None


def extend_negative_data(conver, poolcand, K):
    cands = conver['cand']
    for cand in cands:
        indexs = get_k_random_indices(len(poolcand), K)
        for index in indexs:
            #print ('add candidate: ', poolcand[index])
            #print ('cand 0: ', cand[0])
            cand.append(poolcand[index])
            #sys.exit(1)


def read_training_data(data_path, build_dic=True, extend = 0, extract_pool_cand=False):
    f = open(data_path, 'r')
    convers = []
    profile = []
    question = []
    partner = []
    answer = []
    cands = []
    true_indexs = []
    item_count = {}
    pool_cands = set()
    for line in f:
        temp_line = line.strip()
        if len(temp_line) > 2:
            p_text = extract_text_from_line_numb(temp_line)
            p_text = preprocess_rm_period(p_text)
            if temp_line.startswith('1 your persona'):
                if len(profile) > 0:
                    if len(question) == 0:
                        print ('empty question: ', temp_line)
                    convers.append({'profile': profile, 'question': question, 'answer': answer, 'cand': cands, 'indexs': true_indexs, 'partner': partner})
                temp_p = p_text.split(' ')
                add_count_items(item_count, temp_p)
                profile = [temp_p]
                question = []
                answer = []
                cands = []
                true_indexs = []
                partner = []
            elif is_partner_persona(temp_line):
                temp_p = p_text.split(' ')
                add_count_items(item_count, temp_p)
                partner.append(temp_p)
            else:
                if '\t' not in temp_line:
                    temp_p = p_text.split(' ')
                    add_count_items(item_count, temp_p)
                    profile.append(temp_p)
                else:
                    tgs = p_text.split('\t')
                    item_question = preprocess_rm_period(tgs[0].strip())
                    #print ('question: ', item_question)
                    item_question = item_question.split(' ')
                    add_count_items(item_count, item_question)
                    question.append(item_question)
                    item_answer = preprocess_rm_period(tgs[1].strip())
                    #print ('item answer: ', item_answer)
                    cand_items = tgs[3].split('|')
                    #print ('number of cand_items: ', len(cand_items))
                    t_index = find_true_index(item_answer, cand_items)
                    #print ('index: ', t_index)
                    true_indexs.append(t_index)
                    if t_index == -1:
                        print ('cannot find true answer for this case: %s' % temp_line)
                    item_answer = item_answer.split(' ')
                    add_count_items(item_count, item_answer)
                    answer.append(item_answer)
                    rm_items = []
                    for item in cand_items:
                        temp_item = preprocess_rm_period(item)
                        pool_cands.add(temp_item)
                        temp_item = temp_item.split(' ')
                        add_count_items(item_count, temp_item)
                        rm_items.append(temp_item)
                    cands.append(rm_items)
    f.close()
    vocab = None
    #print ('number of convers from file: %d' % len(convers))
    if build_dic:
        #print (item_count)
        vocab = cut_off_low_frequency(item_count, None, -1)
    poollist = list(pool_cands)
    if extract_pool_cand:
        save_f = open('cand_pool.txt', 'w')
        for line in poollist:
            save_f.write('%s\n'%line)
        save_f.close()
    if extend > 0:
            print ('number of candidate pool: %d' % len(poollist))
            sep_poollist = []
            for item in poollist:
                sep_poollist.append(item.split(' '))
            for conver in convers:
                extend_negative_data(conver, sep_poollist, extend)
    print ('number of cand per question: ', len(convers[0]['cand'][0]))
    return convers, vocab


def cut_off_low_frequency(item_count, threshold = None, max_vocab = 100000):
    result = {UNKNOWN_TOKEN: 1, PADDING_TOKEN: 0}
    if threshold is None and max_vocab == -1:
        for key in item_count:
            result[key] = len(result)
        return result
    if threshold is not None:
        for key in item_count:
            if item_count[key] >= threshold:
                result[key] = len(result)
    else:
        pairs = sorted(item_count.items(), key=lambda x: x[1])
        leng = len(pairs)
        for i in range(len(pairs)):
            index = leng - 1 - i
            key = pairs[index]
            result[key[0]] = len(result)
            if i > max_vocab:
                break
    return result


def get_wids_from_list_sens(vocab_dic, p_sens, max_length):
    p_ids = []
    for sen in p_sens:
        temp_ids = [vocab_dic[tok] if tok in vocab_dic else vocab_dic[UNKNOWN_TOKEN] for tok in sen]
        for i in range(len(sen), max_length):
            temp_ids.append(vocab_dic[PADDING_TOKEN])
        p_ids.append(temp_ids)
    return p_ids


def padding_list_sen_with_leng(sen_inds, vocab_dic, max_leng):
    result = []
    for sen in sen_inds:
        temp_res = list(sen)
        for i in range(len(sen), max_leng):
            temp_res.append(vocab_dic[PADDING_TOKEN])
        result.append(temp_res)
    return result



def extract_data_from_w2vec(vocab_dic):
    from w2vec import redis_lookup
    id2w = {}
    for key in vocab_dic:
        id2w[vocab_dic[key]] = key
    keys = id2w.keys()
    keys.sort()
    v_list = []
    for key in keys:
        word = id2w[key]
        vector = redis_lookup.get_embedded_vecto(word)
        v_list.append(vector)
    matrix = np.array(v_list)
    return matrix


def save_setting(setting, save_path):
    save_json(setting, save_path)


def read_setting(save_path):
    return read_json(save_path)


def read_json(fpath):
    f = open(fpath, 'r')
    text = f.read()
    f.close()
    return json.loads(text)


def save_json(json_data, save_path):
    f = open(save_path, 'w')
    f.write(json.dumps(json_data, indent=4, ensure_ascii=False))
    f.close()

def save_w2vec_matrix(matrix):
    w2vec_path = get_w2vec_path()
    np.savetxt(w2vec_path, matrix)


def get_w2vec_path():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    w2vec_path = os.path.join(current_folder, 'embedding.out')
    return w2vec_path


def load_w2vec_matrix():
    w2vec_path = get_w2vec_path()
    return np.loadtxt(w2vec_path)



def save_vocab(vocab, save_path):
    vocab_str = json.dumps(vocab, indent=4, ensure_ascii=False)
    f = open(save_path, 'w')
    f.write(vocab_str)
    f.close()


def load_vocab(vocab_path):
    return read_json(vocab_path)


def get_vocab_path():
    return os.path.join(CURRENT_FOLDER, 'dictionary.json')


def main():
    data_path = 'data/train_self_original.txt'
    convers, vocab_dic = read_training_data(data_path, True, 20)
    print ('number of conversations: %d' % len(convers))
    print ('start building dictionary: %d words' % len(vocab_dic))
    print ('sample data: ')
    print (json.dumps(convers[0]))
    w2vec_matrix = extract_data_from_w2vec(vocab_dic)
    save_w2vec_matrix(w2vec_matrix)
    save_vocab(vocab_dic, get_vocab_path())


def extract_cand():
    data_path = 'data/train_self_original.txt'
    read_training_data(data_path, True, 0, True)

if __name__ == '__main__':
    extract_cand()
    #main()