model_name=my_model # custom name for the model
openai_key=my_key # OpenAI API key
split=val # choose from val, test
mode=reasoning # choose from reasoning, descriptive
model_path="your_path" # path to the model, customizable argument

python src/generate.py \
    --model_name $model_name \
    --split $split \
    --mode $mode \
    --model_path $model_path

python src/evaluate.py \
    --model_name $model_name \
    --split $split \
    --mode $mode \
    --api_key $openai_key

python src/get_score.py \
    --model_name $model_name \
    --split $split \
    --mode $mode
