#!/bin/bash
cd ../src
# datasets=('PEMS04' 'OEDI-WA' 'OEDI-WI')
# missing_patterns=('block' 'point')

datasets=('PEMS04')
missing_patterns=('block' 'point')
DWT_levels=(2)

for DWT_level in ${DWT_levels[*]}; do
for dataset in ${datasets[*]}; do
for missing_pattern in ${missing_patterns[*]}; do

python_script="main_MFDiff.py"
scratch=True
checkpoint_path=""
cuda='cuda:0'
seq_len=24

if [ "$missing_pattern" == "block" ]; then
    missing_ratio=0.02
else
    missing_ratio=0.25
fi

dataset_path="../datasets/$dataset/"
log_path="../logs/new_$missing_pattern/$dataset"

if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
    echo "Folder created: $log_path"
else
    echo "Folder already exists: $log_path"
fi

for ((i=1; i<=1; i++))  # default 5 exps
do
    seed=$i

    echo "Running iteration $i with seed $seed on device $cuda"

    if [ $scratch = True ]; then
        nohup python -u $python_script \
            --scratch $scratch \
            --device $cuda \
            --seed $seed \
            --dataset $dataset \
            --dataset_path $dataset_path \
            --seq_len $seq_len \
            --missing_pattern $missing_pattern \
            --missing_ratio $missing_ratio \
            --DWT_level $DWT_level \
            > $log_path/${dataset}_${missing_pattern}${missing_ratio}_MFDiff_${DWT_level}levelDWT_seed${seed}_$(date +%Y%m%d%H%M).log 2>&1 &
    else
        nohup python -u $python_script \
            --scratch $scratch \
            --device $cuda \
            --seed $seed \
            --dataset $dataset \
            --dataset_path $dataset_path \
            --seq_len $seq_len \
            --missing_pattern $missing_pattern \
            --missing_ratio $missing_ratio \
            --checkpoint_path $checkpoint_path \
            --nsample 100 \
            --DWT_level $DWT_level \
            > $log_path/${dataset}_${missing_pattern}${missing_ratio}_MFDiff_${DWT_level}levelDWT_seed${seed}_$(date +%Y%m%d%H%M).log 2>&1 &
    fi

    wait

    echo ""
done

done
done
done