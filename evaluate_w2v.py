#!/usr/bin/env python3.7

print('Importing...')
import argparse
import numpy as np
from gensim.test.utils import common_texts, get_tmpfile, datapath
from gensim.models import Word2Vec, KeyedVectors
import os, sys


filename = 'filtered-question-words-fr.txt'
prefix = '/home/fkirefu/meili0/faheem/gpanlp/GloVe/eval/question-data'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    args = parser.parse_args()
    to_save = args.vectors_file.replace('models_w2v', 'models_w2v/eval')+'.eval'
    print(to_save)
    if os.path.exists(to_save):
              
        print("Skipping {}".format(args.vectors_file))
        return None
    print('Loading vector file')
    wv = KeyedVectors.load(args.vectors_file)
    
    words = list(wv.wv.vocab.keys())
 
    print('W')

    W = wv[words]
    print('vocab')
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    print('NORMALISING')
    W_norm = np.divide(W.T, np.linalg.norm(W, ord=2, axis=1)).T

    print('evaluating')
    evaluate_vectors(W_norm, vocab, ivocab, to_save)

def evaluate_vectors(W, vocab, ivocab, to_save):
    """Evaluate the trained word vectors on a variety of tasks"""

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_tot = 0 # count correct questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words

    with open('%s/%s' % (prefix, filename), 'r') as f:
        full_data = [line.rstrip().split(' ') for line in f]
        full_count += len(full_data)
        data = [x for x in full_data if all(word in vocab for word in x)]

    indices = np.array([[vocab[word] for word in row] for row in data])
    ind1, ind2, ind3, ind4 = indices.T

    predictions = np.zeros((len(indices),))
    num_iter = int(np.ceil(len(indices) / float(split_size)))
    for j in range(num_iter):
        print(j, num_iter)
        subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

        pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
            +  W[ind3[subset], :])
        #cosine similarity if input W has been normalized
        dist = np.dot(W, pred_vec.T)

        for k in range(len(subset)):
            dist[ind1[subset[k]], k] = -np.Inf
            dist[ind2[subset[k]], k] = -np.Inf
            dist[ind3[subset[k]], k] = -np.Inf

        # predicted word index
        predictions[subset] = np.argmax(dist, 0).flatten()

    val = (ind4 == predictions) # correct predictions
    count_tot = count_tot + len(ind1)
    correct_tot = correct_tot + sum(val)

    with open(to_save, 'w') as out:
    
        out.write('Questions seen/total: %.2f%% (%d/%d)\n' %
        (100 * count_tot / float(full_count), count_tot, full_count))    
        
        
        out.write('Total accuracy: %.2f%%  (%i/%i)\n' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))
    print('Questions seen/total: %.2f%% (%d/%d)' %
        (100 * count_tot / float(full_count), count_tot, full_count))
    print('Total accuracy: %.2f%%  (%i/%i)' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))


if __name__ == "__main__":
    print('Starting program')
    main()
