#DATA_DIR="../data/abbadingo/Mid"
DATA_DIR="../data/abbadingo/Short_Sequences_Mid"

# train data
for filepath in ${DATA_DIR}/04.*PT.4.*_Train.txt.dat; do
  filename=$(basename "$filepath")
  echo "Doing file ${filename}"
  python ../model_scripts/train_causal_models_short_strings.py ${filename}
done