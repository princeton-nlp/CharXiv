model_name=my_model # custom name for the model
openai_key=my_key # OpenAI API key
split=val # choose from val, test
mode=reasoning # choose from reasoning, descriptive
model_path=my_model_path # path to the model, customizable argument

python generate.py \
    --model_name $model_name \
    --split $split \
    --mode $mode \
    --model_path $model_path

python evaluate.py \
    --model_name $model_name \
    --split $split \
    --mode $mode \
    --api_key $openai_key

python get_score.py \
    --model_name $model_name \
    --split $split \
    --mode $mode
