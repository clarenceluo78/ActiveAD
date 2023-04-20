datasets=('1_ALOI' '12_fault' '17_InternetAds' '19_landsat' '20_letter'\
            '22_magic.gamma')
# prev_datasets=('44_Wilt' '32_shuttle' '31_satimage-2' '2_annthyroid'\
#             '28_pendigits' '38_thyroid' '10_cover'  '26_optdigits' \
#         )
# little_datasets=('45_wine' '14_glass' '18_Ionosphere' '21_Lymphography' '20_letter' \
#             '40_vowels' '37_Stamp' '15_Hepatitis')
budget=(0.05 0.1 0.25 0.5 0.75)
normal_strategies=('RandomSampling' 'LeastConfidence'  'MarginSampling' 'EntropySampling'\
        'LeastConfidenceDropout' 'MarginSamplingDropout' 'EntropySamplingDropout' \
        'BALDDropout' 'VarRatio' 'MeanSTD' 'MetaSampling'
    )
embedding_strategies=('KMeansSampling' 'KMeansSamplingGPU' 'KCenterGreedy' 'KCenterGreedyPCA'\
            'BadgeSampling' 'AdversarialBIM' 'AdversarialDeepFool'
)

for dataset in ${datasets[*]}
do
    echo $dataset
    for budget in ${budget[*]}
    do
        echo $budget
        # # XGBOD
        # model='XGBOD'
        # for strategy in ${normal_strategies[*]}
        # do
        #     echo "dataset: ${dataset}   model: ${model}   strategy: ${strategy}"
        #     python pipeline.py --dataset $dataset --model $model --strategy $strategy --budget $budget
        # done

        # # DevNet
        # model='DevNet'
        # for strategy in ${normal_strategies[*]}
        # do
        #     echo "dataset: ${dataset}   model: ${model}   strategy: ${strategy}"
        #     python pipeline.py --dataset $dataset --model $model --strategy $strategy --budget $budget
        # done
        # for strategy in ${embedding_strategies[*]}
        # do
        # echo "dataset: ${dataset}   model: ${model}   strategy: ${strategy}"
        # python pipeline.py --dataset $dataset --model $model --strategy $strategy --budget $budget
        # done

        # # WAAL
        # model='WAAL'
        # strategy='WAAL'
        # echo "dataset: ${dataset}   model: ${model}   strategy: ${strategy}"
        # python pipeline.py --dataset $dataset --model $model --strategy $strategy --budget $budget

        # LPL
        model='LPL'
        strategy='LossPredictionLoss'
        echo "dataset: ${dataset}   model: ${model}   strategy: ${strategy}"
        python pipeline.py --dataset $dataset --model $model --strategy $strategy --budget $budget
    done
done