balance_ratio=1,1
model_name=roberta
initial_size=300

command="python model_gen.py --balance_ratio $balance_ratio --model_name $model_name --initial_size $initial_size"
echo "Executing command: $command"
$command
