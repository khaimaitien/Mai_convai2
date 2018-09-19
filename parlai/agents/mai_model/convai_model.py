import torch
from torch import nn
import utility
from torch.autograd import Variable
import torch.nn.functional as F


def get_default_setting():
    return {
        'sim_dim': 512,
        'att_type': 'concat',
        'profile_dim': 512,
        'att_dim': 512
    }


class RankingModel(nn.Module):
    def __init__(self, setting):
        nn.Module.__init__(self)
        self.vocab_size = setting['vocab_size']
        self.sim_dim = setting['sim_dim']
        #self.use_cuda = setting['use_cuda']
        self.att_type = setting['att_type']
        self.p_dim = setting['profile_dim']
        if self.att_type == 'concat':
            self.att_dim = setting['att_dim']
            self.linear_att = nn.Linear(self.p_dim + self.sim_dim, self.att_dim)
            ini_fil = torch.randn(self.att_dim)
            abs_norm = torch.norm(ini_fil)
            ini_fil = ini_fil/abs_norm
            ini_fil = ini_fil.unsqueeze(1)
            self.va = nn.Parameter(ini_fil) # self.att_dim * 1
        elif self.att_type == 'matrix':
            self.temp_linear = nn.Linear(self.p_dim, self.p_dim)
        embed_init = None
        if 'embed_init' in setting:
            embed_init = setting['embed_init']
        retrain_emb = setting['retrain_emb']
        word_dim = setting['word_dim']
        self.embed_layer = nn.Embedding(self.vocab_size, word_dim)
        if embed_init is not None:
            self.embed_layer.weight = nn.Parameter(torch.Tensor(embed_init))
        if retrain_emb:
            self.embed_layer.weight.requires_grad = True
        else:
            self.embed_layer.weight.requires_grad = False
        self.profile_item_encoder = nn.LSTM(word_dim, self.p_dim, num_layers=1, batch_first=True)
        self.query_encoder = nn.LSTM(word_dim, self.p_dim, num_layers=1, batch_first=True)
        self.candidate_encoder = nn.LSTM(word_dim, self.sim_dim, num_layers=1, batch_first=True)
        self.linear_sim_dim = nn.Linear(self.p_dim * 2, self.sim_dim)

    def get_init_for_lstm(self, use_cuda, batch_size, lstm_dim):
        first = Variable(torch.zeros(1, batch_size, lstm_dim))
        second = Variable(torch.zeros(1, batch_size, lstm_dim))
        if use_cuda:
            first = first.cuda()
            second = second.cuda()
        return (first, second)


    def get_output_from_lstm(self, use_cuda, lstm_layer, output_dim, input_batch, lengths):
        batch_size = input_batch.size(0)
        init_ve = self.get_init_for_lstm(use_cuda, batch_size, output_dim) #self.init_hidden_query(use_cuda, batch_size)
        output, hidden = lstm_layer(input_batch, init_ve)
        l_output = utility.get_output_at_length(lengths, output, True)
        return l_output


    def encode_profile(self, use_cuda, profile_sens, profile_lengths):
        item_embeded = self.embed_layer(profile_sens)
        return self.get_output_from_lstm(use_cuda, self.profile_item_encoder, self.p_dim, item_embeded, profile_lengths)


    def encode_candidate(self, use_cuda, candidates, cand_lengths):
        k = candidates.size(1)
        N = candidates.size(0)
        flat_answer = candidates.view(k * N, -1)  # view (N*k) * max_a
        e_answer = self.embed_layer(flat_answer)
        # flat a_lengs #
        flat_a_lengs = []
        for item in cand_lengths:
            flat_a_lengs.extend(item)
        v_answer = self.get_output_from_lstm(use_cuda, self.candidate_encoder, self.sim_dim, e_answer, flat_a_lengs)  # (N * K) * sim_dim
        v_answer = torch.tanh(v_answer) #get_norm_l2(v_answer)  # normalize to l2
        v_answer = v_answer.view(N, k, -1)  # N * k * sim_dim
        return v_answer


    def get_profile_rep_by_attention(self, p_encoded, q_encoded):
        """

        :param p_encoded: P * p_dim
        :param q_encoded: N * p_dim
        :return: (N * P) --> score for each profile to attention
        """
        P = p_encoded.size(0)
        N = q_encoded.size(0)
        if self.att_type == 'concat':
            q_dim = q_encoded.size(1)
            temp_q_encoded = q_encoded.unsqueeze(1)  # N * 1 * p_dim
            temp_q_encoded = temp_q_encoded.expand(N, P, q_dim)
            flat_q = temp_q_encoded.contiguous().view((N*P), q_dim)

            p_dim = p_encoded.size(1)
            temp_p_encoded = p_encoded.unsqueeze(0) # 1 * P * p_dim
            temp_p_encoded = temp_p_encoded.expand(N, P, p_dim)
            flat_p = temp_p_encoded.contiguous().view((N*P), p_dim)
            concat_reps = torch.cat([flat_q, flat_p], dim=1) # (N*P) * (2*p_dim)

            con_reps = self.linear_att(concat_reps) # (N*P) * att_dim
            con_reps = torch.tanh(con_reps)# (N*P) * att_dim
            score_reps = torch.mm(con_reps, self.va) #
            score_reps = score_reps.squeeze(1)
            score_reps = score_reps.view(N, P) # N * P
            score_reps = F.softmax(score_reps, dim=1) # N * P
            final_rep = torch.mm(score_reps, p_encoded)# (N, P)  * (P, p_dim) --> N * p_dim
            return final_rep

    def forward(self, use_cuda, profile_sens, profile_lengths, question, question_lengths, candidates, candidate_lengths):
        p_encoders = self.encode_profile(use_cuda, profile_sens, profile_lengths) # P * p_dim
        e_query = self.embed_layer(question) # N * max_leng * word_dim
        v_query = self.get_output_from_lstm(use_cuda, self.query_encoder, self.p_dim, e_query, question_lengths) # N * p_dim
        v_cands = self.encode_candidate(use_cuda, candidates, candidate_lengths) # N * k * sim_dim
        N = v_cands.size(0)
        K = v_cands.size(1)
        sim_dim = v_cands.size(2)
        v_cand_flat = v_cands.view((N * K), sim_dim)
        p_att = self.get_profile_rep_by_attention(p_encoders, v_cand_flat) # (N*K) * p_dim

        # expand v_query ##
        v_query = v_query.unsqueeze(1) # N * 1 * p_dim
        p_dim = v_query.size(2)
        v_query = v_query.expand(N, K, p_dim) # N * K * p_dim
        v_query = v_query.contiguous().view((N*K), p_dim) # (N*K) * p_dim

        comb_context = torch.cat([p_att, v_query], dim=1)
        comb_context = self.linear_sim_dim(comb_context) # (N *k) sim_dim
        comb_context = torch.tanh(comb_context) # (N *K) * sim_dim
        scores = F.cosine_similarity(comb_context, v_cand_flat) # N*K
        scores = scores.view(N, K)
        return scores