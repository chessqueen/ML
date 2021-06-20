INPUT_FILE = "climate_change_health_abstract.txt"
OUTPUT_FILE = 'climate_change_health_document_sentence_vector.csv'


import re
import numpy as np
import pandas as pd
from collections import Counter

with open(INPUT_FILE, 'r', encoding='utf-8') as fh:
    all_data = fh.read()

all_data = all_data.replace("\n\n", 'XXX').replace("\n", ' ').replace("XXX", '\n')

info_pmid = re.split('(PMID: \d+)[^\n]+\n',  all_data)


def get_title(info):
    lines = info.split('\n')
    return lines[1]

def get_abstract(info):
    lines = info.split('\n')
    line_lengths = [len(line) for line in lines] # the longest one is (usually?) the abstract. KLUDGE ALERT!
    return lines[np.argmax(line_lengths)]

document_text = pd.DataFrame([{'pmid':info_pmid[i+1].replace('PMID: ', ''), 'text': get_title(info_pmid[i]) + ' ' + get_abstract(info_pmid[i])} for i in range(0, len(info_pmid) - 1, 2)])

document_text

from spacy.lang.en import English

nlp = English()
nlp.add_pipe('sentencizer')

sentences = [ [sent for sent in nlp(txt).sents] for txt in document_text['text'].values]
sentences

import spacy

document_sentences = [(doc_id, sent_id, str(sent)) for (doc_id, doc) in zip(range(len(sentences)), sentences)
 for (sent_id, sent) in zip(range(len(doc)), doc)]

document_sentence_pdf = pd.DataFrame(document_sentences, columns=['document_id', 'sentence_number', 'sentence'])

sentence_encoder = spacy.load('en_use_md')
document_sentence_pdf['vector'] = [sentence_encoder(s).vector for s in document_sentence_pdf['sentence'].values]

document_sentence_pdf.head()

document_sentence_pdf[['document_id', 'sentence_number']]
