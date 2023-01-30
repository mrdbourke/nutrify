file_list=("train.py" "evaluate.py")

for py_file in "${file_list[@]}"
do
    python ${py_file}
done