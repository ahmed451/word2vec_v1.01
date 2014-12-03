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
#include <string.h>
#include <math.h>
#include <stdlib.h>

const long long max_size = 13000;         // max length of strings
const long long N = 100;                  // number of closest words that will be shown
const long long max_w = 80;              // max length of vocabulary entries
const long long max_wrds = 1300;

int main(int argc, char **argv) {
  FILE *f;
  char st1[max_size];
  char bestw[N][max_wrds];
  char file_name[200], st[max_wrds][max_w];
  float dist, len, bestd[N], vec[max_size];
  long long words, size, a, b, c, d, cn, bi[max_wrds];
  char ch;
  float *M;
  char *vocab;
  
  if (argc !=3) {
    printf("Usage: ./compute-distance <FILE> \"phrase\"\n\tFILE : word projections in the BINARY FORMAT\n\tphrase : setence/word to compute similarity for.\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
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
  //while (1) {
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    //printf("Enter word or sentence (EXIT to break): ");
    a = 0;
    /*while (1) {
      st1[a] = fgetc(stdin);
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;
        break;
      }
      a++;
      
    }*/
    //printf("Reading String...len: %lld\n",strlen(argv[2]));
    strcpy(st1, argv[2]);
    if (!strcmp(st1, "EXIT")) return 0; //break;
    cn = 0;
    b = 0;
    c = 0;
    while (c<strlen(st1)) {
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
	//printf("Read ome more word: %lld - at %lld \"%s\"\n",cn,c,st[cn-1]);
      }
    }
    cn++;
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = -1;
      bi[a] = b;
      //if (b == -1) {
      //  printf("%s: Out of dictionary word! - Ignore!!\n",st[a]);
        //break;
      //}
    }
    //if (b == -1) return 0;//continue;
    //printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n");
    //printf("Computing Similarity ...\n");
    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
      if (bi[b] == -1) continue;
      for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
    }
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (c = 0; c < words; c++) {
      a = 0;
      for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
      if (a == 1) continue;
      dist = 0;
      for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
      for (a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          for (d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          strcpy(bestw[a], &vocab[c * max_w]);
          break;
        }
      }
    }
    // AA. Stop at the first 
    for (a = 0; a < 1; a++) printf("%s\t%f\t%zu\n", bestw[a], bestd[a], strlen(argv[2]));
    // 
    //for (a = 0; a < N; a++) printf("%s\t%f\n", bestw[a], bestd[a]);
  //}
  return 0;
}
