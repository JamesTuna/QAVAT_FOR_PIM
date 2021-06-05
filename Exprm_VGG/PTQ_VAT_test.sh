S_nbits_weight="1 2 3 4"
S_noise="0.1 0.2 0.3 0.4 0.5"
nbits_act=5
mkdir saved_PTQVAT
for nbits_weight in $S_nbits_weight
do
    for noise in $S_noise
    do
        echo "PTQVAT: A${nbits_act}W${nbits_weight} noise $noise"
        python3 PTQ_VAT_test.py --nbits_weight $nbits_weight --nbits_act $nbits_act \
                                --noise $noise --testSize 10000 --testSample 1000 \
                                --cuda 0 --tune_epoch 1
    
    done
done

S_nbits_weight="1 2 3 4"
S_noise="0.1 0.2 0.3 0.4 0.5"
nbits_act=5
mkdir saved_PTQVAT
for nbits_weight in $S_nbits_weight
do
    for noise in $S_noise
    do
        echo "PTQVAT: A${nbits_act}W${nbits_weight} noise $noise"
        python3 PTQ_VAT_test.py --nbits_weight $nbits_weight --nbits_act $nbits_act \
                                --noise $noise --testSize 10000 --testSample 1000 \
                                --cuda 0 --tune_epoch 0
    
    done
done
