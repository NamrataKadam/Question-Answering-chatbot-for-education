from src.components import QueryProcessor, PassageRetrieval, AnswerExtractor
import os, datetime
from werkzeug.utils import secure_filename
import pdfbox
import spacy
import codecs

SPACY_MODEL = os.environ.get('SPACY_MODEL', 'en_core_web_sm')
#QA_MODEL = os.environ.get('QA_MODEL', 'distilbert-base-cased-distilled-squad')
QA_MODEL = os.environ.get('QA_MODEL', 'ktrapeznikov/albert-xlarge-v2-squad-v2')
nlp = spacy.load(SPACY_MODEL, disable=['ner', 'parser', 'textcat'])
query_processor = QueryProcessor(nlp)
passage_retriever = PassageRetrieval(nlp)
answer_extractor = AnswerExtractor(QA_MODEL, QA_MODEL)

def main():
    dict1 = {}
    directory = "E:\\Studies\\DL\\project\\BookQnA\\src\\books\\improved\\files\\"
    count = 0;
    for filename in os.listdir(directory):
        file = directory + filename;
        print("Filename", filename)
        doc = codecs.open(file, 'r', 'UTF-8').read()
        print("document content:----",doc)
        print("---**********************************----")
        passage_retriever.fit(doc)
        qna = codecs.open("E:\\Studies\\DL\\project\\BookQnA\\src\\books\\improved\\qna\\" + filename, 'r',
                          'UTF-8').readlines()
        print(qna)
        right_ans = 0
        i = 0
        length = len(qna)

        result = filename + "\n" + doc + "\n\n"
        while i < length:
            question = qna[i][2:-2]
            print("Question: ", question)
            result = result + "Question: " + question + "\n";
            passages = passage_retriever.most_similar(question)
            answers = answer_extractor.extract(question, passages)
            i += 1;
            if (answers):
                for key, value in answers[0].items():
                    if key == 'answer':
                        #value = value.replace(",", "")
                        #value = value.replace(".", "")
                        #value = value.replace("'s", "")
                        #value = value.replace(")", "")
                        #value = value.replace("\"", "")
                        gr_truth_ans = qna[i][2:-2].split("#")
                        print ("Ground truth ans: ", gr_truth_ans)
                        for g in range(len(gr_truth_ans)):
                            if gr_truth_ans[g].strip() == value.strip():
                                right_ans += 1
                                break
                        print("Ground Truth Answer: ", qna[i][2:-2])
                        result = result + "Ground Truth Answer: " + qna[i][2:-2] + "\n"
                        print("Predicted Answer: ", value)
                        result = result + "Predicted Answer: " + value + "\n\n"
            else:
                if qna[i][2:-2] == "No Answer":
                    print("Ground Truth Answer: ", qna[i][2:-2])
                    result = result + "Ground Truth Answer: " + qna[i][2:-2] + "\n"
                    print("Predicted Answer: <No Answer>")
                    result = result + "Predicted Answer: <No Answer>" + "\n\n"
                    right_ans += 1
                else:
                    print("Ground Truth Answer: ", qna[i][2:-2])
                    result = result + "Ground Truth Answer: " + qna[i][2:-2] + "\n"
                    print("Predicted Answer: <No Answer>")
                    result = result + "Predicted Answer: <No Answer>" + "\n\n"
            i += 1
        total = length / 2
        if total != 0:
            accuracy = right_ans / total
        print("total Questions: ", total)
        print("No. of answers predicted correctly: ", right_ans)
        print("accuracy: ", accuracy)
        result = result + "total Questions: " + str(total) + "\n" + "No. of answers predicted correctly: " + str(
            right_ans) + "\n" + "accuracy: " + str(accuracy) + "\n\n"
        file1 = open("performance_a2.txt", "a")  # append mode
        file1.write(result)
        file1.close()
        dict1[str(count)] = [str(count+1), filename, str(total), str(right_ans), str(accuracy)]
        count+=1
    performance_table = "\n" + " Number  " + " Filename " + " Total Questions "+" Correctly Predicted "+" Accuracy "+"\n"
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format('Number', 'Filename', 'Total_Que','Predicted_correctly','Accuracy'))
    for key, value in dict1.items():
        num, filenm, totalq, right, acc = value
        print("{:<10} {:<20} {:<10} {:<10} {:<10} ".format(num, filenm, totalq, right, acc))
        performance_table = performance_table + num + "    " + filenm + "    " + totalq + "    " + right + "   " + acc + "\n"
        print()
    file2 = open("performance_a2.txt", "a")  # append mode
    file2.write(performance_table)
    file2.close()


if __name__ == "__main__":
    main()
