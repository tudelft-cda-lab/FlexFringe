DATA_DIR="../data/original/Mid"
TARGET_DIR="../data/abbadingo/Mid"

mkdir $TARGET_DIR

# train data
for filepath in ${DATA_DIR}/*.SL.*_Train.txt; do
  filename=$(basename "$filepath")
  echo "Doing file ${filename}"
  python convert_data_to_abbadingo.py ${filepath} ${TARGET_DIR}/${filename}.dat
done

#test data
for filepath in ${DATA_DIR}/*.SL.*SR*.txt; do
  filename=$(basename "$filepath")
  echo "Doing file ${filename}"
  python convert_data_to_abbadingo.py ${filepath} ${TARGET_DIR}/${filename}.dat
done