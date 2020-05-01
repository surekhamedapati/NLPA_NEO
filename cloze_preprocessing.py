# -*- coding: utf-8 -*-
import gzip
import json
import pdb
import operator
import string
import re

STOP_WORDS = frozenset(('a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can',
                        'for', 'from', 'have', 'if', 'in', 'is', 'it', 'may',
                        'not', 'of', 'on', 'or', 'tbd', 'that', 'the', 'this',
                        'to', 'us', 'we', 'when', 'will', 'with', 'yet',
                        'you', 'your', 'inc', 'll', 'i'))

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

@staticmethod
    def cleanText(text):
        """
        Method to map a fixed list of higher frequency category names
        to names that will not be split by a tokenizer.

        :param text: input string
        :return: cleaned string
        """

        transform_list = [(r"'s\b",' '),
                          (r"(^|\W)[.]net(\W|$)","\g<1>dotnet\\2"),
                          (r"(^|\W)c[+][+](\W|$)", "\g<1>cplusplus\\2"),
                          (r"(^|\W)objective[-]c(\W|$)", "\g<1>objectivec\\2"),
                          (r"(^|\W)node[.]js(\W|$)", "\g<1>nodedotjs\\2"),
                          (r"(^|\W)asp[.]net(\W|$)", "\g<1>aspdotnet\\2"),
                          (r"(^|\W)e[-]commerce(\W|$)", "\g<1>ecommerce\\2"),
                          (r"(^|\W)java[-]ee(\W|$)", "\g<1>javaee\\2"),
                          (r"(^|\W)32[-]bit(\W|$)", "\g<1>32bit\\2"),
                          (r"(^|\W)kendo[-]ui(\W|$)", "\g<1>kendoui\\2"),
                          (r"(^|\W)jquery[-]ui(\W|$)", "\g<1>jqueryui\\2"),
                          (r"(^|\W)c[+][+]11(\W|$)", "\g<1>cplusplus11\\2"),
                          (r"(^|\W)windows[-]8(\W|$)", "\g<1>windows8\\2"),
                          (r"(^|\W)ip[-]address(\W|$)", "\g<1>ipaddress\\2"),
                          (r"(^|\W)backbone[.]js(\W|$)", "\g<1>backbonedotjs\\2"),
                          (r"(^|\W)angular[.]js(\W|$)", "\g<1>angulardotjs\\2"),
                          (r"(^|\W)as[-]if(\W|$)", "\g<1>asif\\2"),
                          (r"(^|\W)actionscript[-]3(\W|$)", "\g<1>actionscript3\\2"),
                          (r"(^|\W)[@]placeholder(\W|$)", "\g<1>atsymbolplaceholder\\2"),
                          (r"\W+", " ")
                          ]
        return_text = text.lower()
        for p in transform_list:
            return_text = re.sub(p[0], p[1], return_text)

        return return_text

def prepCloze():

    file_contexts = ['data/dev_contexts.json.gz', 'data/test_contexts.json.gz', 'data/train_contexts.json.gz']
    file_questions = ['data/dev_questions.json.gz', 'data/test_questions.json.gz', 'data/train_questions.json.gz']

    filenames_w = ['data/dev.json', 'data/test.json', 'data/train.json']
    #file_testing = ['data/dev_testing.txt', 'data/test_testing.txt']

    for i in range(3):
        count = 0
        fpr_c = gzip.open(file_contexts[i], 'r')
        fpr_q = gzip.open(file_questions[i], 'r')
        fpw = open(filenames_w[i], 'w')
        if i != 2:
            fpw_testing = open(file_testing[i], 'w')
        for line in fpr_c:
            context = json.loads(line)
            question = json.loads(fpr_q.readline())
            assert(context["uid"] == question["uid"])
            fpw.write(question["question"]+'\t'+question["uid"]+'\t'+question["answer"]+'\n')
            for c in context['contexts']:
                fpw.write(c[1]+'\t'+str(c[0])+'\n')
            #fpw.write('*new_instance*\n')
            if i != 2:
                fpw_testing.write(question["uid"]+'\t'+question["answer"]+'\n')

        fpr_c.close()
        fpr_q.close()
        fpw.close()

        if i != 2:
            fpw_testing.close()

    print ('preprossing finished!')

if __name__ == "__main__":
    prepCloze()