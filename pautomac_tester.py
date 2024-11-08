"""
[Written by Sicco Verwer]
"""

from numpy   import *
from decimal import *
from sys import *
from pickle import *
import math

def number(arg):
    return Decimal(arg)

def normalize(arr):
 sumarr = number(sum(arr))
 if sumarr != 0.0:
     for i in range(len(arr)):
         arr[i] = number(arr[i]) / sumarr
 else:
     for i in range(len(arr)):
         arr[i] = number(1.0) / number(len(arr))

def perplexity(trueprobs, testprobs):
   sumt = number(0.0)
   log2 = math.log10(number(2.0))
   for index in range(len(trueprobs)):
       if trueprobs[index] == 0.0:
          print("solution zero")
          trueprobs[index] = number(1.0E-200)
       if testprobs[index] == 0.0:
          print("test zero")
          testprobs[index] = number(1.0E-200)

       term = trueprobs[index] * Decimal( math.log10(testprobs[index]) / log2 )
       sumt = sumt + term
   return math.pow(number(2.0),-sumt)

def maxnorm(trueprobs, testprobs):
   sumt = number(0.0)
   for index in range(len(trueprobs)):
       term = abs(trueprobs[index] - testprobs[index])
       sumt = max(sumt,term)
   return sumt

def sumnorm(trueprobs, testprobs):
   sumt = number(0.0)
   for index in range(len(trueprobs)):
       term = abs(trueprobs[index] - testprobs[index])
       sumt = sumt + term
   return sumt

def readprobs(f):
 probs = []
 line = f.readline()
 num_probs = int(line)
 for n in range(num_probs):
 #for line in f:
    line = f.readline()
    #value = line.split(";")[5].strip()
    probs = probs + [number(line)]
 return probs

def readlogprobs(f):
 probs = []
 line = f.readline()
 #num_probs = int(line)
 #for n in range(num_probs):
 #for line in f:
 while True: 
    line = f.readline()
    if not line:
        break
    value = line.split(";")[4].strip()
    probs = probs + [math.exp(number(value))]
 return probs

if __name__ == "__main__":

  getcontext().prec = 250

  if len(argv) != 3:
    print("required input: model solution answer")
    assert(False)

  #model_file = argv[1]
  solution_file = argv[1]
  answer_file = argv[2]

  solution = readprobs(open(solution_file,"r"))
  result   = readlogprobs(open(answer_file,"r"))

  if sum(result != 1.0):
      normalize(result)
  if sum(solution != 1.0):
      normalize(solution)


  #for i in range(10):
  #    print(result[i]);
  #    print(solution[i]);

  perp = perplexity(solution, result)
  perp_true = perplexity(solution, solution)
  maxn = maxnorm(solution, result)
  sumn = sumnorm(solution, result)

  #zeros = 0.0
  #values = 0.0
  #for i in range(len(S)):
  #   for j in range(len(S[i])):
  #      if S[i][j] > 0.0:
  #         values = values + 1.0
  #      if S[i][j] == 0.0:
  #         zeros = zeros + 1.0

  #symbolsparsity = values / (values + zeros)

  #zeros = 0.0
  #values = 0.0
  #for i in range(len(T)):
  #   for j in range(len(T[i])):
  #      for k in range(len(T[i][j])):
  #         if T[i][j][k] > 0.0:
  #            values = values + 1.0
  #         if T[i][j][k] == 0.0:
  #            zeros = zeros + 1.0

  #transitionsparsity = values / (values + zeros)

  #sparsity = 0.5 * (symbolsparsity + (4.0 * transitionsparsity))

  #print "file, perplexity, solution perplexity, size, alphabet, sparsity"
  print(answer_file, float(perp), float(perp_true), float(perp-perp_true))