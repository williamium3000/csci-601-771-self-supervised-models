learning_rate=(1e-4 5e-4 1e-3)
models=("t5-base" "t5-large")
epochs=(5 7 9)

for model in ${models[@]}
do
for lr in ${learning_rate[@]}
do
for epoch in ${epochs[@]}
do
sh mysrun_ncsa.sh $model $lr $epoch 64 &
done
done
done