import os
import pdfbox
import spacy
import codecs
from flask import Flask, render_template, jsonify, request

from src.components import QueryProcessor, PassageRetrieval, AnswerExtractor

app = Flask(__name__)
SPACY_MODEL = os.environ.get('SPACY_MODEL', 'en_core_web_sm')
QA_MODEL = os.environ.get('QA_MODEL', 'distilbert-base-cased-distilled-squad')
nlp = spacy.load(SPACY_MODEL, disable=['ner', 'parser', 'textcat'])
query_processor = QueryProcessor(nlp)
passage_retriever = PassageRetrieval(nlp)
answer_extractor = AnswerExtractor(QA_MODEL, QA_MODEL)


@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST','GET'])
def bookUpload():
    data1 = request.get_json()
    print(data1)
    if(request.method == 'POST'):
        file = data1.get('file');
        print("file---",file);
    #p = pdfbox.PDFBox()
    #p.extract_text('books\class-6-History.pdf')
    return render_template('qa.html')

@app.route('/answer-question', methods=['POST','GET'])
def analyzer():
    if (request.method == 'POST'):
        data = request.get_json()
        question = data.get('question')
        query = query_processor.generate_query(question)
        #file1 = open("books\class-6-History.txt", "r+")
        #doc = "Hello world \nThis is Namrata \n Namrata is software engineer"
        doc = codecs.open('books/class-6-History.txt','r', 'UTF-8').read()
        #file1.close()
        #docs = document_retriever.search(query)
        passage_retriever.fit(doc)
        passages = passage_retriever.most_similar(question)
        answers = answer_extractor.extract(question, passages)
        return jsonify(answers)
    else:
        return render_template('qa.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
