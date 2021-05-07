nbits_act=1
nbits_weight=1
lr="0.001"
epochs=91
decay_ep=30
S_noise="0.0 0.2 0.4 0.6 0.8 1.0"
valSample=1
valSize=10000
mkdir saved
for noise in $S_noise
do
  echo "A${nbits_act}W${nbits_weight} noise $noise"
  python3 train.py --nbits_act $nbits_act --nbits_weight $nbits_weight \
                    --noise $noise  --valSize $valSize --valSample $valSample \
                    --lr $lr --epochs $epochs --decay_ep $decay_ep
done
