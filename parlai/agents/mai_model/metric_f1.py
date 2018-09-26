import re
import data_reader
from collections import Counter
import conversation

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(guess, answers):
    """Check if guess is a (normalized) exact match with any answer."""
    if guess is None or answers is None:
        return False
    guess = normalize_answer(guess)
    for a in answers:
        if guess == normalize_answer(a):
            return True
    return False


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Computes precision, recall and f1 given a set of gold and prediction items.

    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values

    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _f1_score(guess, answers):
    """Return the max F1 score between the guess and *any* answer."""
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split())for a in answers
    ]
    return max(f1 for p, r, f1 in scores)


def eval_f1_for_valid(cand_num, convers):
    total_f1 = 0.0
    count = 0
    for conver in convers:
        profile = conver['profile']
        questions = conver['question']
        answers = conver['answer']
        for index in range(len(questions)):
            question = questions[index]
            answer = answers[index]
            response = conversation.get_response(profile, question, cand_num)
            #print ('response = ', response)
            #print ('answer = ', answer)
            f1 = _f1_score(response, [' '.join(answer)])
            total_f1 += f1
        if count % 10 == 1:
            print ('cand_num = %d, count = %d' % (cand_num, count))
        count += 1
    return total_f1


def find_optimal_cand_num():
    cand_nums = [20, 50, 80, 100, 120, 150, 180, 200]
    convers, _ = data_reader.read_training_data('data/valid_self_original.txt', True, 0)
    print ('leng of converse: %d' % len(convers))
    save_f = open('cand_report.txt', 'w')
    for cand_num in cand_nums:
        result = eval_f1_for_valid(cand_num, convers[: 100])
        print ('cand_num = %d has fscore = %f' % (cand_num, result))
        save_f.write('%d: %f\n' % (cand_num, result))
    save_f.close()


if __name__ == '__main__':
    find_optimal_cand_num()


