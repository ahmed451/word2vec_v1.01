//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100

const int vocab_hash_size = 500000000;  // Maximum 500M entries in the vocabulary

const long long max_size = 250;         // max vector size
const long long max_line_size = 3000;   // max length of strings
const long long max_nlines = 5000000;   // Maximum 5M lines in the reference text
const long long max_w = 80;   

typedef float real;                     // Precision of float numbers

struct vocab_word {
  long long cn;
  char *word;
};

char ref_file[MAX_STRING], test_file[MAX_STRING], vec_file[MAX_STRING];
struct vocab_word *vocab;
int debug_mode = 2, min_count = 5, *vocab_hash, min_reduce = 1;
int binary = 0;
long long vocab_max_size = 10000, vocab_size = 0;
long long train_words = 0;
real threshold = 100;
int N = 1;                // number of closest words that will be shown

unsigned long long next_random = 1;

// Reads a single line from a file
void ReadLine(char *line, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == '\n')) {
      if (a > 0) {
        break;
      }
      if (ch == '\n') {
        strcpy(line, (char *)"</s>");
        return;
      } else continue;
    }
    if(a<=max_line_size)
      line[a] = ch;
    a++;
  }
  if(a<=max_line_size)
    line[a] = 0; 
  else
    line[max_line_size]=0;
}

int EvalModule() {
  long long cn = 0;
  long long besti[N];
  float dist, len, bestd[N],  ph_vec[max_size];
  float *vec; //vec[max_nlines][max_size];
  FILE *f, *fin, *ftst;
 
  char st[max_size][max_size];
  char st1[max_size];
  char strphrase[max_size];
  
  long long words, size, nlines, a, b, c, d, bi[max_size];
  char ch;
  float *M;
  char *vocab;

  char line[max_line_size];

  printf("Starting loading data files ....\n");

  // Read vector file
  printf("Read vec file :%s\n",vec_file );
  f = fopen(vec_file, "rb");
  if (f == NULL) {
    printf("Word vectors/projections file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  vocab_hash = (int *)malloc(words * size * sizeof(int)); // vocab_hash_siz
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    fscanf(f, "%s%c", &vocab[b * max_w], &ch);
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);

  a = 0;

  // Reaf reference file
  vec = (float *)malloc((long long)max_nlines * (long long)size * sizeof(float));
  if (vec == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)max_nlines * size * sizeof(float) / 1048576, max_nlines, size);
    return -1;
  }
  nlines = 0;
  printf("Read reference file :%s\n",ref_file );
  fin = fopen(ref_file, "rb");
  if (fin == NULL) {
    printf("Reference file not found\n");
    return -1;
  }
  while (!feof(fin)) {
      ReadLine(line, fin);
      if(strlen(line)>=max_line_size) {
          printf("WARNING: line %lld reached max_line_size!\n",nlines);
      }
      //printf("Retrieved line %s", line);
      strcpy(st1, line);
      cn = 0;
      b = 0;
      c = 0;
      while (1) {
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
        }
      }
      cn++;
      for (a = 0; a < cn; a++) {
        for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
        if (b == words) b = -1;
        bi[a] = b;
      }
      for (a = 0; a < size; a++) vec[nlines*size + a] = 0;
      for (b = 0; b < cn; b++) {
        if (bi[b] == -1) continue;
        for (a = 0; a < size; a++) vec[nlines* size + a] += M[a + bi[b] * size];
      }
      len = 0;
      for (a = 0; a < size; a++) len += vec[nlines*size + a] * vec[nlines*size + a];
      len = sqrt(len);
      //printf("V%lld : ",nlines);
      for (a = 0; a < size; a++) { 
          vec[nlines*size + a] /= len;
          //printf("%lf ", vec[nlines][a]);
      }
      //printf("\n");
      nlines++;
  }

  fclose(fin);

// Read test file and compare each line
printf("Read test file :%s\n",test_file );
nlines = 0;
ftst = fopen(test_file, "rb");
if (ftst == NULL) {
    printf("Test file not found\n");
    return -1;
  }
  while (!feof(ftst)) {
      ReadLine(line, ftst);

      if(strlen(line)>=max_line_size) {
          printf("WARNING: line %lld reached max_line_size!\n",nlines);
      }
       //printf("Retrieved line %s", line);
  strcpy(strphrase, line);
  //printf("%s - \n",  line);
  nlines++;
  cn = 0;
  b = 0;
  c = 0;
  while (1) {
    st[cn][b] = strphrase[c];
    b++;
    c++;
    st[cn][b] = 0;
    if (strphrase[c] == 0) break;
    if (strphrase[c] == ' ') {
      cn++;
      b = 0;
      c++;
    }
  }
  cn++;
  for (a = 0; a < cn; a++) {
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = -1;
    bi[a] = b;
  }

  // Build the vector for the line
  for (a = 0; a < size; a++) ph_vec[a] = 0;
    for (b = 0; b < cn; b++) {
      if (bi[b] == -1) continue;
      for (a = 0; a < size; a++) ph_vec[a] += M[a + bi[b] * size];
    }
  len = 0;

  // Find the closest/similar reference line number 
  for (a = 0; a < size; a++) len += ph_vec[a] * ph_vec[a];
  len = sqrt(len);
  for (a = 0; a < size; a++) ph_vec[a] /= len;
    for (a = 0; a < N; a++) bestd[a] = 0;
      for (a = 0; a < N; a++) besti[a] = 0;
        for (c = 0; c < nlines; c++) {
          a = 0;
          for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
            if (a == 1) continue;
          dist = 0;
          for (a = 0; a < size; a++) dist += ph_vec[a] * vec[c*size+ a];
            for (a = 0; a < N; a++) {
              if (dist > bestd[a]) {
                for (d = N - 1; d > a; d--) {
                  bestd[d] = bestd[d - 1];
                  besti[d] = besti[d - 1];
                }
                bestd[a] = dist;
                besti[d] = c+1;
                break;
              }
          }
        }
  // AA. Stop at the first 
  for (a = 0; a < N; a++) printf("%lld\t%lld\t %f\n", nlines,besti[a], bestd[a]);

  }

  fclose(ftst);
  return 0;
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

void usage(char * prg) {
    printf("PHRASE2VEC tool v0.1\n");
    printf("Options:\n");
    printf("Parameters :\n");
    printf("\t-wvec <file>\n");
    printf("\t\tWord projections file -in binary format-\n");
    printf("\t-ref <file>\n");
    printf("\t\tUse text from <file> as a reference to compare against\n");
    printf("\t-test <file>\n");
    printf("\t\tUse lines in <file> to evaluate againt ref based on word vectors\n");
    printf("\t-n num\n");
    printf("\t\tNumber of top similar sentences\n");
    printf("\nExamples:\n");
    printf("%s -wvec vectors.bin -ref phrases.txt -test text.txt\n\n", prg);
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1 || argc > 9 ) {
    usage(argv[0]);
   return 0;
  }
  if ((i = ArgPos((char *)"-ref", argc, argv)) > 0) strcpy(ref_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-wvec", argc, argv)) > 0) strcpy(vec_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-test", argc, argv)) > 0) strcpy(test_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-n", argc, argv)) > 0) N = atoi(argv[i + 1]);
  if(N>100) N=100;
  if(EvalModule()<0)
      printf("\n----------------\n");
      usage(argv[0]);

  return 0;
}
