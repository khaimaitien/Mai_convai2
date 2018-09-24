import torch
import data_reader
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


def convert_converse_to_wid(vocab_dic, conver):
    result = {}

    result['profile_leng'] = [len(sen) for sen in conver['profile']]
    p_max_leng = max(result['profile_leng'])
    result['profile'] = data_reader.get_wids_from_list_sens(vocab_dic, conver['profile'], p_max_leng)

    result['question_leng'] = [len(sen) for sen in conver['question']]
    q_max_leng = max(result['question_leng'])
    result['question'] = data_reader.get_wids_from_list_sens(vocab_dic, conver['question'], q_max_leng)

    if 'answer' in conver:
        result['answer_leng'] = [len(sen) for sen in conver['answer']]
        a_max_leng = max(result['answer_leng'])
        result['answer'] = data_reader.get_wids_from_list_sens(vocab_dic, conver['answer'], a_max_leng)
    cand_lengths = []
    total_max_leng = 0
    for cand in conver['cand']:
        temp_cand_length = [len(item) for item in cand]
        total_max_leng = max(total_max_leng, max(temp_cand_length))
        cand_lengths.append(temp_cand_length)
    cands = []
    for cand in conver['cand']:
        temp_cand = data_reader.get_wids_from_list_sens(vocab_dic, cand, total_max_leng)
        cands.append(temp_cand)
    result['cand'] = cands
    result['cand_leng'] = cand_lengths
    if 'indexs' in conver:
        result['index'] = conver['indexs']
    return result

def get_random_of_list(l):
    numb = len(l)
    per = np.random.permutation(numb)
    result = [l[index] for index in per]
    return result

def get_norm_l2_of_vector(x):
    last_dim = x.size(1)
    batch_size = x.size(0)
    x_norm = torch.norm(x, p=2, dim=1)
    x_norm = x_norm.unsqueeze(1)
    x_norm = x_norm.expand(batch_size, last_dim)
    return x / x_norm


def get_output_at_length(input_lengths, output, batch_first):
    idx = (torch.LongTensor(input_lengths) - 1).view(-1, 1).expand(
        len(input_lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    idx = idx.unsqueeze(time_dimension)
    if output.is_cuda:
        idx = idx.cuda(output.data.get_device())
    last_output = output.gather(
        time_dimension, Variable(idx)).squeeze(time_dimension)
    return last_output


def get_loss(use_cuda, scores, indexs, lamda):
    k = scores.size(1)
    N = scores.size(0)
    true_arr = autograd.Variable(torch.zeros(N))
    if use_cuda:
        true_arr = true_arr.cuda()
    for i in range(len(indexs)):
        index = indexs[i]
        #print ('type of scores: ', type(scores[i][index]))
        true_arr[i] = scores[i][index]
    true_pre = 0
    _, max_indexs = torch.max(scores, 1) # N
    max_indexs = max_indexs.data.tolist()
    for i in range(len(indexs)):
        if max_indexs[i] == indexs[i]:
            true_pre += 1

    true_arr = torch.unsqueeze(true_arr, 1)
    true_arr = true_arr.expand(-1, k)
    if use_cuda:
        true_arr = true_arr.cuda()
    loss_m = lamda + scores - true_arr
    loss_m = F.relu(loss_m)
    return torch.sum(loss_m), true_pre


def get_loss_from_converse(conver, use_cuda, model, lamda, just_forward=False):
    scores = get_prediction_from_converse(conver, use_cuda, model, just_forward)
    true_indexs = conver['index']
    loss, true_pre_count = get_loss(use_cuda, scores, true_indexs, lamda)
    return loss, true_pre_count


def get_prediction_from_converse(conver, use_cuda, model, just_forward=False):
    profile = conver['profile']
    var_profile = autograd.Variable(torch.LongTensor(profile))  # N *
    if use_cuda:
        var_profile = var_profile.cuda()
    question_length = conver['question_leng']
    question = conver['question']
    var_question = autograd.Variable(torch.LongTensor(question))
    if use_cuda:
        var_question = var_question.cuda()
    cands = conver['cand']
    var_cand = autograd.Variable(torch.LongTensor(cands))
    if use_cuda:
        var_cand = var_cand.cuda()
    cand_lengths = conver['cand_leng']
    if just_forward:
        with torch.no_grad():
            scores = model(use_cuda, var_profile, conver['profile_leng'], var_question, question_length, var_cand, cand_lengths)
    else:
        scores = model(use_cuda, var_profile, conver['profile_leng'], var_question, question_length, var_cand,
                       cand_lengths)
    return scores
