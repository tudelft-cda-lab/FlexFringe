#DATA_DIR="../data/abbadingo/Mid"
DATA_DIR="../data/abbadingo/Short_Sequences"

# train data
for filepath in ${DATA_DIR}/16.16.SL.2.1.9*.txt.dat; do
  filename=$(basename "$filepath")
  echo "Doing file ${filename}"
  python ../model_scripts/train_causal_models.py ${filename}
done