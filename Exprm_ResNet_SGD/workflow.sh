lr="0.1"
epochs=241
decay_ep=60
S_noise="0 0.1 0.2 0.3 0.4 0.5"
valSample=1
valSize=10000
model='ResNet18'
nbits_act=8
nbits_weight=4

mkdir saved

#echo "${model} FP32 noise 0"
#python3 train.py --noise 0  --valSize $valSize --valSample $valSample \
#                --lr $lr --epochs $epochs --decay_ep $decay_ep \
#                --model $model

for noise in $S_noise
do
  echo "${model} A${nbits_act}W${nbits_weight} noise $noise"
  #echo "${model} FP32 noise $noise"
  python3 train.py --noise $noise  --valSize $valSize --valSample $valSample \
                    --lr $lr --epochs $epochs --decay_ep $decay_ep \
                    --model $model --nbits_act $nbits_act --nbits_weight $nbits_weight
done

nbits_act=8
nbits_weight=2
for noise in $S_noise
do
  echo "${model} A${nbits_act}W${nbits_weight} noise $noise"
  #echo "${model} FP32 noise $noise"
  python3 train.py --noise $noise  --valSize $valSize --valSample $valSample \
                    --lr $lr --epochs $epochs --decay_ep $decay_ep \
                    --model $model --nbits_act $nbits_act --nbits_weight $nbits_weight
done
nbits_act=8
nbits_weight=3
for noise in $S_noise
do
  echo "${model} A${nbits_act}W${nbits_weight} noise $noise"
  #echo "${model} FP32 noise $noise"
  python3 train.py --noise $noise  --valSize $valSize --valSample $valSample \
                    --lr $lr --epochs $epochs --decay_ep $decay_ep \
                    --model $model --nbits_act $nbits_act --nbits_weight $nbits_weight
done
