DATA_DIR="../data/abbadingo/Mid"

# train data
for filepath in ${DATA_DIR}/*.TLT.*_Train.txt.dat; do
  filename=$(basename "$filepath")
  echo "Doing file ${filename}"
  python ../model_scripts/train_distilbert.py ${filename}
done