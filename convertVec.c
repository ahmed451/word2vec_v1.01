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
#include <ctype.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40



const long long max_size = 2000;         // max length of strings
const long long N = 1;                   // number of closest words
const long long max_w = 50; 
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

void usage();
char input_file[MAX_STRING], output_file[MAX_STRING];
char format[MAX_STRING];

//struct vocab_word *vocab;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0, *syn1, *syn1neg, *expTable;

int debug = 0;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;

int ConvBinary2Text() {
  FILE *fin, *fout;
  float *M;
  char *vocab;
  char ch;
  float len;
  long long words, size, a, b;

  fin = fopen(input_file, "rb");
  if (fin == NULL) {
    printf("Input file not found\n");
    return -1;
  } 
  fout = fopen(output_file, "wb");
  if (fout == NULL) {
    printf("Output file not found\n");
    return -1;
  } 
  if(!strcmp(format, "b2t")) {
    fscanf(fin, "%lld", &words);
    fscanf(fin, "%lld", &size);
    fprintf(fout, "%lld %lld\n", words, size);
    if(debug)
     printf("%lld %lld\n", words, size);
  
    vocab = (char *)malloc(words * max_w * sizeof(char));
    M = (float *)malloc(words * size * sizeof(float));
    if (M == NULL) {
      printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
      return -1;
    }
    if(debug)
      printf("Starting using output file %s\n", output_file);
    for (b = 0; b < words; b++) {
      fscanf(fin, "%s%c", &vocab[b * max_w], &ch);
      if(debug)
        printf("Read : %s\n",&vocab[b * max_w]);
      for (a = 0; a < max_w; a++) vocab[b * max_w + a] = tolower(vocab[b * max_w + a]);
      fprintf(fout, "%s ", &vocab[b * max_w]);
      for (a = 0; a < size; a++) {
        fread(&M[a + b * size], sizeof(float), 1, fin);
        fprintf(fout, "%lf ", M[a + b * size]);
      }
      fprintf(fout, "\n");
      len = 0;
      for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
      len = sqrt(len);
      for (a = 0; a < size; a++) M[a + b * size] /= len;
    }
  } else
  if(!strcmp(format, "t2b")) {
    fscanf(fin, "%lld", &words);
    fscanf(fin, "%lld", &size);
    fprintf(fout, "%lld %lld\n", words, size);
    if(debug)
    printf("%lld %lld\n", words, size);

    vocab = (char *)malloc(words * max_w * sizeof(char));
    M = (float *)malloc(words * size * sizeof(float));
    if (M == NULL) {
      printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
      return -1;
    }
    if(debug)
      printf("Starting using output file %s\n", output_file);
    for (b = 0; b < words; b++) {
      fscanf(fin, "%s%c", &vocab[b * max_w], &ch);
  
      if(debug)
        printf("Read : %s\n",&vocab[b * max_w]);
      for (a = 0; a < max_w; a++) vocab[b * max_w + a] = tolower(vocab[b * max_w + a]);
      fprintf(fout, "%s ", &vocab[b * max_w]);
      for (a = 0; a < size; a++) {
        fscanf(fin, "%f%c", &M[a + b * size], &ch);
        //fread(&M[a + b * size], sizeof(float), 1, fin);
        fwrite(&M[a + b * size], sizeof(float), 1, fout);
        //fprintf(fout, "%lf ", M[a + b * size]);
      }
      fprintf(fout, "\n");
      len = 0;
      for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
      len = sqrt(len);
      for (a = 0; a < size; a++) M[a + b * size] /= len;
    }
    if(debug)
      printf("Done!!!\n");
  } else {

  usage();
  } 
  fclose(fin);
  fclose(fout);
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

void usage() {
    printf("Convert vectors fromat between binary and text\n\n");
    printf("Options:\n");
    printf("\t-format [b2t|t2b]\n");
    printf("\t\tb2t: convert binary to text\n");
    printf("\t\tt2b: convert text to binary\n");   
    printf("\t-input inputfile\n");
    printf("\t-output outputfile\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode 0,1 (default = 0 = more info during convertion)\n");
    printf("\nExamples:\n");
    printf("./convertVec -format b2t -intput vec.bin -output vec.txt\n\n");
}
int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    usage();
    return 0;
  }
  output_file[0] = 0;
  if ((i = ArgPos((char *)"-format", argc, argv)) > 0) strcpy(format ,argv[i + 1]);
  if ((i = ArgPos((char *)"-input", argc, argv)) > 0) strcpy(input_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug = atoi(argv[i + 1]);

 i = ConvBinary2Text();
  return 0;
}