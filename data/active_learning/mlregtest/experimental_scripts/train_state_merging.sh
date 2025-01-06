# TODO: currently adjusted to be run from project root directory

FLEXFRINGE_DIR="."

DATA_DIR="data/active_learning/mlregtest/data/abbadingo/Short_Sequences_Mid"
INI_FILE="ini/count_types.ini"
PREDICT_INI_FILE="ini/predict-count-driven.ini"

# train data
#for filepath in ${DATA_DIR}/04.04.PT.2.*_Train.txt.dat; do
for filepath in ${DATA_DIR}/04.04.PT.6.*_Train.txt.dat; do
  filename=$(basename "$filepath")
  echo "Doing file ${filename}"
  ./${FLEXFRINGE_DIR}/flexfringe --ini=${FLEXFRINGE_DIR}/${INI_FILE} ${filepath}
done

# predict data
#for filepath in ${DATA_DIR}/04.04.PT.2.*_TestSR.txt.dat; do
for filepath in ${DATA_DIR}/04.04.PT.6.*_TestSR.txt.dat; do
  filename=$(basename "$filepath")
  echo "Doing file ${filename}"

  APTAFILE="${filepath/_TestSR/_Train}.ff.final.json"
  echo "Aptafile ${APTAFILE}"
  ./${FLEXFRINGE_DIR}/flexfringe --ini=${PREDICT_INI_FILE} --aptafile=${APTAFILE} ${filepath}
done