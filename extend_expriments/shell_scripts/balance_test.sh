# 固定参数
initial_size=300
query_size=300
method_setting=MNAL
subset_size=3000
model_name=bert
# balance_ratio=1,1
# balance_ratio=4,1
# balance_ratio=1,4
# balance_ratio=9,1
balance_ratio=1,9
# 循环从 1 到 10
for start_from_run in {1..10}
do
    # 构建命令
    command="python balance.py --initial_size $initial_size --query_size $query_size --method_setting $method_setting --model_name $model_name --start_from_run $start_from_run --start_from_step 0 --subset_size $subset_size --balance_ratio $balance_ratio"
    
    # 执行命令
    echo "Executing command: $command"
    $command
done