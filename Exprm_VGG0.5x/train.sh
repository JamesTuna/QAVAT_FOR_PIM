nbits_act=8
nbits_weight=1
lr="0.001"
epochs=181
decay_ep=60
S_noise="0.0 0.5 0.3 0.2 0.4 0.1"
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

nbits_act=8
nbits_weight=3
lr="0.001"
epochs=181
decay_ep=60
S_noise="0.0 0.5 0.3 0.2 0.4 0.1"
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

nbits_act=8
nbits_weight=4
lr="0.001"
epochs=181
decay_ep=60
S_noise="0.0 0.5 0.3 0.2 0.4 0.1"
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

nbits_act=5
nbits_weight=5
lr="0.001"
epochs=181
decay_ep=60
S_noise="0.0 0.5 0.3 0.2 0.4 0.1"
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
