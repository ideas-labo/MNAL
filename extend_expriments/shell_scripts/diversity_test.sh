initial_size=300
query_size=700
method_setting=MNAL_div
subset_size=3000
balance_ratio=1,1
# 循环从 1 到 10
for start_from_run in {1..10}
do
    # 构建命令
    command="python diversity_test.py --initial_size $initial_size --query_size $query_size --method_setting $method_setting --start_from_run $start_from_run --start_from_step 2 --subset_size $subset_size --balance_ratio $balance_ratio"
    
    # 执行命令
    echo "Executing command: $command"
    $command
done
