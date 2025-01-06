DATA_DIR="../data/original/Short_Sequences_Mid"
TARGET_DIR="../data/abbadingo/Short_Sequences_Mid"

mkdir $TARGET_DIR

# train data
for filepath in ${DATA_DIR}/04.04.PT.4.*_Train.txt; do
  filename=$(basename "$filepath")
  echo "Doing file ${filename}"
  python convert_data_to_abbadingo.py ${filepath} ${TARGET_DIR}/${filename}.dat
done

#test data
for filepath in ${DATA_DIR}/04.04.PT.4.*SR*.txt; do
  filename=$(basename "$filepath")
  echo "Doing file ${filename}"
  python convert_data_to_abbadingo.py ${filepath} ${TARGET_DIR}/${filename}.dat
done