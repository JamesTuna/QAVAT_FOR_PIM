nbits_act=8
nbits_weight=1
S_noise="0.1 0.2 0.3 0.4 0.5"
model="ResNet18"

for noise in $S_noise
do

	echo "A${nbits_act}W${nbits_weight} noise $noise baseline model"
	python3 test.py --model ${model} --load saved/${model}_A${nbits_act}W${nbits_weight}_noise0.0.ckpt \
			--noise $noise --testSize 10000 --testSample 1000 \
			--nbits_weight $nbits_weight --nbits_act $nbits_act --cuda 0

	echo "noise $noise QAVAT model"
	python3 test.py --model ${model} --load saved/${model}_A${nbits_act}W${nbits_weight}_noise${noise}.ckpt \
			--noise $noise --testSize 10000 --testSample 1000 \
			--nbits_weight $nbits_weight --nbits_act $nbits_act --cuda 0
done

nbits_act=8
nbits_weight=4
S_noise="0.1 0.2 0.3 0.4 0.5"
model="ResNet18"

for noise in $S_noise
do

	echo "A${nbits_act}W${nbits_weight} noise $noise baseline model"
	python3 test.py --model ${model} --load saved/${model}_A${nbits_act}W${nbits_weight}_noise0.0.ckpt \
			--noise $noise --testSize 10000 --testSample 1000 \
			--nbits_weight $nbits_weight --nbits_act $nbits_act --cuda 0

	echo "noise $noise QAVAT model"
	python3 test.py --model ${model} --load saved/${model}_A${nbits_act}W${nbits_weight}_noise${noise}.ckpt \
			--noise $noise --testSize 10000 --testSample 1000 \
			--nbits_weight $nbits_weight --nbits_act $nbits_act --cuda 0
done

nbits_act=8
nbits_weight=2
S_noise="0.1 0.2 0.3 0.4 0.5"
model="ResNet18"

for noise in $S_noise
do

	echo "A${nbits_act}W${nbits_weight} noise $noise baseline model"
	python3 test.py --model ${model} --load saved/${model}_A${nbits_act}W${nbits_weight}_noise0.0.ckpt \
			--noise $noise --testSize 10000 --testSample 1000 \
			--nbits_weight $nbits_weight --nbits_act $nbits_act --cuda 0

	echo "noise $noise QAVAT model"
	python3 test.py --model ${model} --load saved/${model}_A${nbits_act}W${nbits_weight}_noise${noise}.ckpt \
			--noise $noise --testSize 10000 --testSample 1000 \
			--nbits_weight $nbits_weight --nbits_act $nbits_act --cuda 0
done

nbits_act=8
nbits_weight=3
S_noise="0.1 0.2 0.3 0.4 0.5"
model="ResNet18"

for noise in $S_noise
do

	echo "A${nbits_act}W${nbits_weight} noise $noise baseline model"
	python3 test.py --model ${model} --load saved/${model}_A${nbits_act}W${nbits_weight}_noise0.0.ckpt \
			--noise $noise --testSize 10000 --testSample 1000 \
			--nbits_weight $nbits_weight --nbits_act $nbits_act --cuda 0

	echo "noise $noise QAVAT model"
	python3 test.py --model ${model} --load saved/${model}_A${nbits_act}W${nbits_weight}_noise${noise}.ckpt \
			--noise $noise --testSize 10000 --testSample 1000 \
			--nbits_weight $nbits_weight --nbits_act $nbits_act --cuda 0
done