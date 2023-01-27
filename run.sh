
#python mordred_pretrain.py --p_dname chembl --p_seed 27407 --backbone gin
# python mordred_pretrain.py --p_dname zinc --p_seed 27407 --backbone gin
python mordred_pretrain.py --p_dname chembl+zinc --p_seed 27407 --backbone gin


for data_id in 1 2
do
    for split_id in {0..2}
    do
        for train_size_id in {0..6}
        do
            for pretrain_mode in mordred_chembl+zinc_gin
            do
    
                python run_code.py --data_id $data_id --split_id $split_id --train_size_id $train_size_id --pretrain_mode $pretrain_mode --pretraining_epoch 10
            done
         done
    done

done

for r_seed in {27407..27409}
do
    for split_id in {1..4}
    do
        for pretrain_mode in mordred_chembl+zinc_gin
        do
            python run_code.py --data_id 3 --split_id $split_id --train_size_id $split_id --pretrain_mode $pretrain_mode --r_seed $r_seed --pretraining_epoch 10
        done
    done
done
