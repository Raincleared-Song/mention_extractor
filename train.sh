python main.py -t supervised -m train
srun -G 1 -c 4 --mem 16g python3 main.py -t supervised -m train
python main.py -t fewshot -m train -c checkpoint/fewnerd-mention_bio-bert_crf-inter0505/model/step-60000.pkl --part inter --n_way 5 --n_shot 5
python main.py -t fewshot -m train -c checkpoint/fewnerd-mention_bio-bert_crf-inter1001/model/step-12000.pkl --part inter --n_way 10 --n_shot 1
python main.py -t fewshot -m train -c checkpoint/fewnerd-mention_bio-bert_crf-intra0505/model/step-110000.pkl --part intra --n_way 5 --n_shot 5
python main.py -t fewshot -m train -c checkpoint/fewnerd-mention_bio-bert_crf-intra1001/model/epoch-0.pkl --part intra --n_way 10 --n_shot 1


python main.py -t fewshot -m train -c checkpoint/fewnerd-mention_bio-bert_crf-inter0505_200/model/step-10000.pkl --part inter --n_way 5 --n_shot 5
python main.py -t fewshot -m train -c checkpoint/fewnerd-mention_bio-bert_crf-inter1001_200/model/step-10000.pkl --part inter --n_way 10 --n_shot 1
python main.py -t fewshot -m train -c checkpoint/fewnerd-mention_bio-bert_crf-intra0505_200/model/epoch-1 --part intra --n_way 5 --n_shot 5
python main.py -t fewshot -m train -c checkpoint/fewnerd-mention_bio-bert_crf-intra1001_200/model/step-10000.pkl --part intra --n_way 10 --n_shot 1


python main.py -t supervised -m train -tc checkpoint/fewnerd-mention_bio-bert_crf-pretrain_base/model/step-160000-a100.pkl --device cuda:0
python main.py -t supervised -m train --device cuda:7 -c checkpoint/fewnerd-mention_bio-bert_crf-pretrain_t5_large/model/step-160000.pkl --resume
python main.py -t supervised -m train -tc checkpoint/fewnerd-mention_bio-bert_crf-pretrain_base_sf_noinit/model/step-140000.pkl --device cuda:5 \
  -c checkpoint/fewnerd-mention_bio-bert_crf-pretrain_base_sf_noinit/model/step-140000.pkl --resume
python main.py -t supervised -m train -tc checkpoint/fewnerd-mention_bio-bert_crf-pretrain_base_sf_reinit/model/step-140000.pkl --device cuda:6 \
  -c checkpoint/fewnerd-mention_bio-bert_crf-pretrain_base_sf_reinit/model/step-140000.pkl --resume
