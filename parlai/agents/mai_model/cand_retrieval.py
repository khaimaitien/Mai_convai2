from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
import os, datetime

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INDEX_FOLDER = os.path.join(CURRENT_FOLDER, 'indexdir')

def read_stop_word():
    stopword_path = os.path.join(CURRENT_FOLDER, 'stopwords.txt')
    result = set()
    f = open(stopword_path, 'r')
    for line in f:
        temp_line = line.strip()
        if len(temp_line) >= 1:
            result.add(temp_line)
    f.close()
    return result

STOP_WORDS = read_stop_word()
print ('number of stopwords: %d' % len(STOP_WORDS))
PUNCT_SET = set('!,.?;')

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

def get_index():
    schema = Schema(question=TEXT(stored=True))
    create_folder(INDEX_FOLDER)
    ix = create_in(INDEX_FOLDER, schema)
    return ix


M_INDEX = get_index()


def index_text_from_file(file_path):
    t1 = datetime.datetime.now()
    f = open(file_path, 'r')
    writer = M_INDEX.writer()
    did = 0
    for line in f:
        temp_line = line.strip()
        if len(temp_line) > 0:
            writer.add_document(question=temp_line)
            did += 1
            if did % 100000 == 1:
                print ('count = %d' % did)
    f.close()
    print ('indexing: %d sentences' % did)
    writer.commit()
    t2 = datetime.datetime.now()
    print ('time for indexing: %f seconds' % (t2 - t1).total_seconds())



def get_documents(sample_sentence, limit = 50):
    token_text = sample_sentence
    tgs = token_text.split(' ')
    query_p = ' OR '.join(tgs)
    with M_INDEX.searcher() as searcher:
        query = QueryParser("question", M_INDEX.schema).parse(query_p)
        search_results = searcher.search(query, limit=limit)
        cands = []
        #print ('leng = %d'%len(search_results))
        for item in search_results:
            cands.append(item['question'])
        return cands
    return []

index_text_from_file(os.path.join(CURRENT_FOLDER, 'cand_pool.txt'))


def main():
    while True:
        input_str = input('input: ')
        cands = get_documents(input_str)
        print (cands)


if __name__ == '__main__':
    main()
