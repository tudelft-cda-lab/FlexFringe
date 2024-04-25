export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/" # I need this one for FST
cd ../data/languages/fst
for file in ./*.fst
do 
  echo "${file}.dot"
  fstdraw --portrait $file >> "${file}.dot"
  dot -Tpdf "${file}.dot" >> "${file}.pdf"
  dot -Tjpg "${file}.dot" >> "${file}.jpg"
  rm "${file}.dot"
done

#cd ../data/languages/fst
#for file in ./*.fst
#do 
#  echo "${file}.dot"
#  fstdraw --portrait $file >> "${file}.dot"
  #dot -Tpdf "${file}.dot" >> "${file}.pdf"
  #dot -Tjpg "${file}.dot" >> "${file}.jpg"
  #rm "${file}.dot"
#done