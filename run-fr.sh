
time ./word2vec -train ../data/sent.uniq.fr -output vectors.bin -cbow 1 -size 128 -window 8 -negative 25 -hs 1 -sample 1e-4 -threads 20 -binary 1 -iter 15 -save-vocab vocab-from-w2c.txt
./distance vectors.bin
