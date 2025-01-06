
DATA_DIR="../data/abbadingo/Short_Sequences_Mid"

# train data
for filepath in ${DATA_DIR}/04.04.PT.6.1.*TestSR.txt.dat; do
#for filepath in ${DATA_DIR}/04.03.TSL.6.*TestSR.txt.dat; do
#for filepath in ${DATA_DIR}/04.04.PT.2.1.1_*TestSR.txt.dat; do
  filename=$(basename "$filepath")
  echo "Doing file ${filename}"

  #PROBLEM=( $(IFS="_" echo "${filename}") )
  IFS='_' read -ra PROBLEM <<< "$filename"
  python evaluate_sm_model.py ${PROBLEM[0]} ${DATA_DIR} --min_test_length=0 --output_dir=state_merging_larger
done