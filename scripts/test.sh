list=(1 2 3 4 5)
list2=(6 7 8 9 10)
for i in ${list[@]}
do
    for j in ${list2[@]}
    do
        echo $i $j
    done
done