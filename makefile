CC = gcc
#CC = clang
#The -Ofast might not work with older versions of gcc; in that case, use -O2
#CFLAGS = -lm -g3 -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result
CFLAGS = -lm -g3 -pthread -O0 -march=native -Wall -funroll-loops -Wno-unused-result
#CFLAGS = -lm -pthread -O2 -march=native -Wall -funroll-loops -Wunused-result


all: word2vec phrase2vec word2phrase distance convertVec compute-distance word-analogy compute-accuracy readproj

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
phrase2vec : phrase2vec.c
	$(CC) phrase2vec.c -o phrase2vec $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
compute-distance : compute-distance.c
	$(CC) compute-distance.c -o compute-distance $(CFLAGS)
readproj : readproj.c
	$(CC) readproj.c -o readproj $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
	chmod +x *.sh
convertVec : convertVec.c
	$(CC) convertVec.c -o convertVec $(CFLAGS)

clean:
	rm -rf word2vec phrase2vec word2phrase distance convertVec compute-distance word-analogy compute-accuracy readproj
