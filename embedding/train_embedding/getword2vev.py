from gensim.models import word2vec
import sys

def train_w2v(sentences, train_mode, output_path):
    print("train", train_mode, "model")
    _sg = (train_mode=="skipgram")
    model = word2vec.Word2Vec(sentences, size=300, min_count=5, workers=8, sg=_sg)
    print("save", train_mode, "model...")
    model.wv.save_word2vec_format("%s.%s.txt" % (output_path, train_mode), binary=False)


if __name__ == "__main__":
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print("create sentences...")
    sentences = word2vec.LineSentence(input_path)

    train_w2v(sentences, "cbow", output_path)
    train_w2v(sentences, "skipgram", output_path)