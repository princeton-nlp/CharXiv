# 📊 CharXiv
🏠[Home](https://charxiv.github.io/) | 🤗[Data](https://huggingface.co/datasets/princeton-nlp/CharXiv) | 🥇[Leaderboard](https://charxiv.github.io/#leaderboard) | 📄Paper (Soon!) | *Current Version: Beta*

This repository contains the code to evaluate models on CharXiv from the paper CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs.

*🤗 We are first-time evaluation suite builders and this codebase is released for the first-time. We are committed to improving it. If you have any questions, feel free to raise issues and/or submit pull requests for new features of bug fixes.*

### 🔊 An 80-second video teaser of CharXiv

https://github.com/princeton-nlp/CharXiv/assets/59942464/ab9b293b-8fd6-4735-b8b3-0079ee978b61

## 👋 Introduction
Chart understanding plays a pivotal role when applying Multimodal Large Language Models (MLLMs) to real-world tasks such as analyzing scientific papers or financial reports. However, existing datasets often focus on oversimplified and homogeneous charts with template-based questions, leading to an over-optimistic measure of progress. In this work, we propose CharXiv, a comprehensive evaluation suite involving 2,323 natural, challenging, and diverse charts from scientific papers. CharXiv includes two types of questions: (1) descriptive questions about examining basic chart elements and (2) reasoning questions that require synthesizing information across complex visual elements in the chart. To ensure quality, all charts and questions are handpicked, curated, and verified by human experts. Our results reveal a substantial, previously underestimated gap between the reasoning skills of the strongest proprietary model (i.e., GPT-4o), which achieves 47.1% accuracy, and the strongest open-source model (i.e., InternVL Chat V1.5), which achieves 29.2%. All models lag far behind human performance of 80.5%, underscoring weaknesses in the chart understanding capabilities of existing MLLMs. We hope CharXiv facilitates future research on MLLM chart understanding by providing a more realistic and faithful measure of progress.

![Comparison of model performance](https://charxiv.github.io/static/images/comparison.png)

## 🛠️ Evaluate Your Models
### Setup
Download the images:
```
git clone https://github.com/princeton-nlp/CharXiv.git
cd images
wget https://huggingface.co/datasets/princeton-nlp/CharXiv/resolve/main/images.zip
unzip images.zip && rm images.zip
```

<details>
<summary>  (Optional) A short tour for the codebase </summary>

```
.
├── data/
│   ├── descriptive_test.json
│   ├── descriptive_val.json
│   ├── image_metadata_test.json
│   ├── image_metadata_val.json
│   ├── reasoning_test.json
│   ├── reasoning_val.json
│   ├── README.md
│   └── LICENSE
├── images/
│   ├── 0.jpg
│   ├── ...
│   ├── 2399.jpg
│   └── README.md
├── results/
│   └── README.md
├── src/
│   ├── constants.py
│   ├── descriptive_utils.py
│   ├── reasoning_utils.py
│   ├── evaluate.py
│   ├── generate.py
│   └── get_score.py
├── run.sh
├── README.md
├── LICENSE
└── .gitignore
```
* `data` folder contains all QAs and metadata for images, descriptive questions, and reasoning questions. Answers for the test split are intentionally made to `null` to prevent testing data from leaking into the public.  
* `images` folder contains all images where their identifiers range from 0 to 2399. Note that there are only 2333 images in total and the numberings are **not** consecutive.  
* `results` folder contains all response generation and scoring results.  
* `src` folder contains all python code for CharXiv:  
  * `constants.py` stores all the prompts and mappings from question ids to actual questions.  
  * `descriptive_utils.py` contains all code to build queries for response generation and grading, as well as saving all artifacts for descriptive questions.  
  * `reasoning_utils.py` contains all code to build queries for response generation and grading, as well as saving all artifacts for reasoning questions.  
  * `evaluate.py` is the main function to evaluate model responses against the answer with gpt API calls.  
  * `generate.py` is the main function to loop QAs for model to generate responses.  
  * `get_score.py` is the main function to print the reasoning and descriptive question scores.
* `run.sh` is the script to evaluate models


</details>

### Response generation
CharXiv doesn't require any third-party python library when prompting your models to generate responses to the chart-question pairs. Therefore, to set up your model, you should implement the `custom_evaluate` function in `generate.py`. Specifically, this function takes `queries` as the input, which contain all the charts and questions CharXiv uses to evaluate models. It has the following structure:
```js
{
    figure_id:{
        'question': ...<str>
        'figure_path': ...<str>
    },
    ...
    figure_id:{
        'question': ...<str>
        'figure_path': ...<str>
    },
}
```
Once you load your models and all preprocessing functions, simply implement the `evaluate` function in `src/generate.py`:
```py
for k in tqdm(queries):
    query = queries[k]['question']
    image = queries[k]["figure_path"]
    query, image = preprocess(query, image) #TODO
    response = model.chat(query, image) #TODO
    queries[k]['response'] = response
```

To generate model responses:
```bash
python generate.py \
   --model_name $model_name \
   --split $split \
   --mode $mode \
   --model_path $model_path #custom arg
```
⚠️**The first three arguments are required and you should not delete them**. It is your responsibility to ensure the correctness and integrity of the evaluation pipeline if you change them. In particular,

* `--model_name` is used as a string identifier to name the resulting file and you can use whatever string except "-", which we reserve to parse file names to get meta information.

* You should either choose `val` or `test` for the `--split` argument. To perform end-to-end in-house evaluations, you should use `val` we do not release answers for `test`.

* Finally, you should either choose `descriptive` or `reasoning` for the `--mode` argument.

✅The last argument i.e., `--model_path` is a custom argument and feel free to delete it, modify it or add more args as you see fit.

🗄️ Once the process ends, you will find a file in `results` folder named:
`gen-<model_name>-<mode>_<split>.json`. This file stores your model's responses.

### Evaluation

⚠️ Make sure the openai python API library is installed in your evaluation environment. If not, `pip install openai` first.

```bash
python evaluate.py \
    --model_name $model_name \
    --split $split \
    --mode $mode \
    --api_key $openai_key
```

The first three arguments are same as the above (response generation), except that you need to provide an openai API key to evaluate your model's responses.

This python script will automatically match the `gen-<model_name>-<mode>_<split>.json` file in the `results` folder and the `<mode>_<split>.json` file in the `data` folder.

🗄️ Once the process ends, you will find a file in results folder named: `scores-<model_name>-<mode>_<split>.json`. This file stores your model's evaluation results.

Finally, run this:
```bash
python get_score.py \
    --model_name $model_name \
    --split $split \
    --mode $mode \
```
This python script will automatically match the `scores-<model_name>-<mode>_<split>.json` file in the `results` folder to calculate the score and print the score in your console.

Note: we will be updating the `get_score.py` file soon to include more metrics (i.e., metrics we reported in our paper).

## 📜 License
Our original data contributions (all data except the charts) are distributed under the [CC BY-SA 4.0](data/LICENSE) license. Our code is licensed under [Apache 2.0](LICENSE) license. The copyright of the charts belong to the original authors, where you can find the source in `image_metadata_val.json` and `image_metadata_test.json` under the data folder.


## 🥺 Cite
If you use our work and are inspired by our work, please consider cite us (available soon):
```
```
## 🙌 Contributors and Acknowledgement
**📊 CharXiv is built by a team consisting of:**  
Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, Alexis Chevalier, Sanjeev Arora, Danqi Chen

Princeton Language and Intelligence, Princeton University  
University of Wisconsin, Madison  
The University of Hong Kong.

🤗 We adapted part of the [MathVista](https://github.com/lupantech/MathVista)'s codebase in building our evaluation framework, and we greatly appreciate their contributions to the MLLM community.  
🤗 The lyrics in the teaser video are created by GPT-4o from our abstract, and the music is created by Suno. Video is manually edited using CapCut.
