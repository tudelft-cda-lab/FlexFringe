#DATA_DIR="../data/abbadingo/Mid"
DATA_DIR="../data/abbadingo/Short_Sequences_Mid"

# train data
for filepath in ${DATA_DIR}/04.04.PT.4.1.*.txt.dat; do
  filename=$(basename "$filepath")
  echo "Doing file ${filename}"
  python ../model_scripts/test_causal_model.py ${filename}
done