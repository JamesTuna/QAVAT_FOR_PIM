nbits_act=8
nbits_weight=2
S_noise="0.0"
mkdir saved
for noise in $S_noise
do

	echo "noise $noise baseline model"
	python3 test.py --load saved/VGG_A${nbits_act}W${nbits_weight}_noise0.0.ckpt \
			--noise $noise --testSize 10000 --testSample 1000 \
			--nbits_weight $nbits_weight --nbits_act $nbits_act --cuda 0

	echo "noise $noise QAVAT model"
	python3 test.py --load saved/VGG_A${nbits_act}W${nbits_weight}_noise${noise}.ckpt \
			--noise $noise --testSize 10000 --testSample 1000 \
			--nbits_weight $nbits_weight --nbits_act $nbits_act --cuda 0
done
