datasets=('44_Wilt' '32_shuttle' '31_satimage-2' '2_annthyroid'\
            '28_pendigits' '38_thyroid' '10_cover'  '26_optdigits' \
        )
for dataset in ${datasets[@]};
do
    python pipeline.py --dataset $dataset --model DevNet --strategy MetaSampling
done