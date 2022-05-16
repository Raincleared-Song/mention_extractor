python main.py -t supervised -m train
srun -G 1 -c 4 --mem 16g python3 main.py -t supervised -m train
python main.py -t fewshot -m train -c checkpoint/fewnerd-mention_bio-bert_crf-inter0505/model/step-60000.pkl
python main.py -t fewshot -m train -c checkpoint/fewnerd-mention_bio-bert_crf-inter1001/model/step-12000.pkl
python main.py -t fewshot -m train -c checkpoint/fewnerd-mention_bio-bert_crf-intra0505/model/step-110000.pkl
python main.py -t fewshot -m train -c checkpoint/fewnerd-mention_bio-bert_crf-intra1001/model/epoch-0.pkl
