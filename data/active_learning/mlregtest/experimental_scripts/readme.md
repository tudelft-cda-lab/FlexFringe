To solve problem with OpenFST:
- export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"
- Above problem was solved with https://github.com/kaldi-asr/kaldi/issues/3009
- Then, fstdraw 04.03.TLT.2.1.0.fst >> 0.test.dot
- Then dot -Tpdf 0.test.dot >> 00.test.pdf