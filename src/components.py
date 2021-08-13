import operator

from gensim.summarization.bm25 import BM25
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline


class QueryProcessor:

    def __init__(self, nlp, keep=None):
        self.nlp = nlp
        self.keep = keep or {'PROPN', 'NUM', 'VERB', 'NOUN', 'ADJ'}

    def generate_query(self, text):
        doc = self.nlp(text)
        query = ' '.join(token.text for token in doc if token.pos_ in self.keep)
        return query


class PassageRetrieval:

    def __init__(self, nlp):
        self.tokenize = lambda text: [token.lemma_ for token in nlp(text)]
        self.bm25 = None
        self.passages = None

    def preprocess(self, doc):
        passages = [p for p in doc.split('\n') if p and not p.startswith('=')]
        #for x in range(0, len(passages)-5):
            #passages[x: x+5] = [''.join(passages[x: x+5])]
            #del passages[x+1: x+6]
        print("number of passages: ", len(passages))
        print("\n", passages)
        return passages

    def fit(self, docs):
        passages = self.preprocess(docs)
        corpus = [self.tokenize(p) for p in passages]
        self.bm25 = BM25(corpus)
        self.passages = passages

    def most_similar(self, question, topn=10):
        tokens = self.tokenize(question)
        scores = self.bm25.get_scores(tokens)
        pairs = [(s, i) for i, s in enumerate(scores)]
        pairs.sort(reverse=True)
        passages = [self.passages[i] for _, i in pairs[:topn]]
        return passages


class AnswerExtractor:

    def __init__(self, tokenizer, model):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        model = AutoModelForQuestionAnswering.from_pretrained(model)
        self.nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)

    def extract(self, question, passages):
        answers = []
        for passage in passages:
            try:
                answer = self.nlp(question=question, context=passage)
                answer['text'] = passage
                answers.append(answer)
            except KeyError:
                pass
        answers.sort(key=operator.itemgetter('score'), reverse=True)
        return answers
