for i in {1..48}; do echo $i; ./flexfringe --ini ini/cms.ini data/PAutomaC-competition_sets/$i.pautomac.train.dat; done
for i in {1..48}; do echo $i; ./flexfringe --ini ini/predict-cms.ini --aptafile=data/PAutomaC-competition_sets/$i.pautomac.train.dat.ff.final.json data/PAutomaC-competition_sets/$i.pautomac.test.dat; done
rm perplexities.txt
for i in {1..48}; do python pautomac_tester.py data/PAutomaC-competition_sets/$i.pautomac_solution.txt data/PAutomaC-competition_sets/$i.pautomac.train.dat.ff.final.json.result >> perplexities.txt ; done
