nbits_act=8
nbits_weight=1
S_noise="0.1 0.2 0.3 0.4 0.5"
mkdir saved
for noise in $S_noise
do

	echo "noise $noise baseline model"
	python3 test.py --load VGG_A${nbits_act}W${nbits_weight}_noise0.0.ckpt \
			--noise $noise --testSize 10000 --testSample 1000 \
			--nbits_weight $nbits_weight --nbits_act $nbits_act --cuda 0

	echo "noise $noise QAVAT model"
	python3 test.py --load VGG_A${nbits_act}W${nbits_weight}_noise${noise}.ckpt \
			--noise $noise --testSize 10000 --testSample 1000 \
			--nbits_weight $nbits_weight --nbits_act $nbits_act --cuda 0
done
