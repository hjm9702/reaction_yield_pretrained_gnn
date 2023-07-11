python main_pretrain.py --pca_dim 40 --seed 27407

python main_finetune.py --data_id 1 --split_id 0 --train_size_id 0
python main_finetune.py --data_id 2 --split_id 0 --train_size_id 0
python main_finetune.py --data_id 3 --split_id 1 --train_size_id 1 --seed 27407