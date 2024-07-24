# 📊 CharXiv
🏠[Home](https://charxiv.github.io/) | 🤗[Data](https://huggingface.co/datasets/princeton-nlp/CharXiv) | 🥇[Leaderboard](https://charxiv.github.io/#leaderboard) | 📄[Paper](https://arxiv.org/abs/2406.18521) | *Current Version: v1.0*

This repository contains the code to evaluate models on CharXiv from the paper [CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs](https://arxiv.org/abs/2406.18521).

*🤗 We are first-time evaluation suite builders and this codebase is released for the first-time. We are committed to improving it. If you have any questions, feel free to raise issues and/or submit pull requests for new features of bug fixes.*

### 🔊 An 80-second video teaser of CharXiv

https://github.com/princeton-nlp/CharXiv/assets/59942464/ab9b293b-8fd6-4735-b8b3-0079ee978b61

## 📰 News
**[07/26/2024]** 🚀 Upcoming this week: we'll be releasing scores for [GPT-4o-mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) as well as the largest and most capable open-weight VLM in our benchmark: [InternVL2 LLaMA-3 76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B). Alongside scores, we find some interesting patterns in the **trend of model improvement with respect to differnet chart understanding benchmarks** on X. Stay tuned!  
**[07/24/2024]** 🚀 We released the [full evaluation pipeline](https://github.com/princeton-nlp/CharXiv) (i.e., v1.0).  
**[07/23/2024]** 🚀 We released our [evaluation results](https://huggingface.co/datasets/princeton-nlp/CharXiv/tree/main/existing_evaluations) on **all 34 MLLMs** that we have tested so far -- this includes all models' responses to CharXiv's challenging questions, scores graded by GPT-4o, as well as aggregated stats.   
**[07/14/2024]** 🚀 We further evaluated the latest [InternVL Chat V2.0 26B](https://huggingface.co/OpenGVLab/InternVL2-26B) and [Cambrian 34B models](https://huggingface.co/nyu-visionx/cambrian-34b) on CharXiv with some **State-of-the-Art results**. More analysis are [here](https://x.com/zwcolin/status/1812650435808792731).

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
* `data` folder contains all QAs and metadata for images, descriptive questions, and reasoning questions. Answers for the test split are intentionally made to `null` to prevent testing data from leaking into the public.  
* `images` folder contains all images where their identifiers range from 0 to 2399. Note that there are only 2333 images in total and the numberings are **not** consecutive.  
* `results` folder contains all response generation and scoring results.  
* `src` folder contains all python code for CharXiv:  
  * `constants.py` stores all the prompts and mappings from question ids to actual questions.  
  * `descriptive_utils.py` contains all code to build queries for response generation and grading, as well as saving all artifacts for descriptive questions.  
  * `reasoning_utils.py` contains all code to build queries for response generation and grading, as well as saving all artifacts for reasoning questions.  
  * `evaluate.py` is the main function to evaluate model responses against the answer with gpt API calls.  
  * `generate.py` is the main function to loop QAs for model to generate responses.  
  * `get_stats.py` is the main function to print the reasoning and descriptive question statistics.
  * `generate_lib` contains a series of implementations that enable one to generate response on their models.
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

To run CharXiv for your model, go to `src/generate_lib/` directory and create a new python file e.g.,`custom.py`. You can put whatever auxiliary code (e.g., model split function, image processing function, etc) inside this file, but you need to implement the `generate_response` function which takes in `model_path` and `queries` by default. Inside the function, you load your models and all preprocessing functions, and let your model generate responses. The below script is a simple example of usage. If you need more references, you can look at existing `*.py` files how these functions are implemented for different models.


```py
for k in tqdm(queries):
    query = queries[k]['question'] # This will be a single question with instructions
    image = queries[k]["figure_path"] # This will be the path to the figure associated with the above query
    query, image = preprocess(query, image) #TODO
    response = model.chat(query, image) #TODO
    queries[k]['response'] = response
```

Once you finish implementing the `generate_response` function, go to `src/generate_lib/utils.py` and modify the `get_generate_fn` to include the function you implemented. Say if your model's checkpoint folder is called `custom_1` and you implement the `generate_response` function in the `custom.py` file, then  all you need to do it to add the follwing code:

```py
...
elif model_name in ['custom_1']:
    from .custom import generate_response
...
```

Then, to generate model responses:
```bash
python src/generate.py \
    --model_name $model_name \
    --split $split \
    --mode $mode \
    --model_path $model_path 
```
⚠️**The first three arguments are required and you should not delete them**. It is your responsibility to ensure the correctness and integrity of the evaluation pipeline if you change them. In particular,

* `--model_name` is used as a string identifier to name the resulting file and you can use whatever string except "-", which we reserve to parse file names to get meta information.

* You should either choose `val` or `test` for the `--split` argument. To perform end-to-end in-house evaluations, you should use `val` we do not release answers for `test`.

* Finally, you should either choose `descriptive` or `reasoning` for the `--mode` argument.

✅The last argument i.e., `--model_path` is a custom argument and feel free to delete it, modify it or add more args as you see fit. Correspondingly, you should consider changing the input argument of your `generate_response` function and the function that calls `generate_response` if you change `--model_path`.

🗄️ Once the process ends, you will find a file in `results` folder named:
`gen-<model_name>-<mode>_<split>.json`. This file stores your model's responses.

Note: if you are evaluating a model that is **hosted on cloud and can only be accessed via an API**, the `--model_path` argument will correspond to the name of the model e.g., `gpt-4o-2024-05-13`. Also, in creating the custom file in the `src/generate_lib` directory, you need to implement an additional function i.e., `get_client_model` that takes in the `model_path` argument and the `api_key` argument. In addition, you need to add another `elif` statement in `get_client_fn` inside `src/generate_lib/utils.py` with instructions similar to the above. Specific instructions to implement `get_client_model` function differ by API providers, and examples are provided in `gpt.py`, `gemini.py`, `reka.py`, `claude.py`, and `qwen.py`.

### Evaluation

⚠️ Make sure the openai python API library is installed in your evaluation environment. If not, `pip install openai` first.

```bash
python src/evaluate.py \
    --model_name $model_name \
    --split $split \
    --mode $mode \
    --api_key $openai_key
```

The first three arguments are same as the above (response generation), except that you need to provide an openai API key to evaluate your model's responses.

This python script will automatically match the `gen-<model_name>-<mode>_<split>.json` file in the `results` folder and the `<mode>_<split>.json` file in the `data` folder.

🗄️ Once the process ends, you will find a file in results folder named: `scores-<model_name>-<mode>_<split>.json`. This file stores your model's evaluation results graded by LLMs.

Finally, run this:
```bash
python src/get_stats.py \
    --model_name $model_name \
    --split $split 
```
This python script will automatically match the `scores-<model_name>-<mode>_<split>.json` file in the `results` folder to calculate the stats for aggregated scores. Calling this function will automatically generate `stats-<model_name>-<mode>_<split>.json` in the `results` folder.

## 📅 Results from Existing Models
We release full results on the validation set (i.e., generated responses, grading done by LLMs and the aggregated stats) for all models we tested in our [HuggingFace Repo](https://huggingface.co/datasets/princeton-nlp/CharXiv/tree/main/existing_evaluations). If you are interested in doing some fine-grained analysis on these results or calculate some customized metrics, feel free to use them.

## 📜 License
Our original data contributions (all data except the charts) are distributed under the [CC BY-SA 4.0](data/LICENSE) license. Our code is licensed under [Apache 2.0](LICENSE) license. The copyright of the charts belong to the original authors, where you can find the source in `image_metadata_val.json` and `image_metadata_test.json` under the data folder.


## 🥺 Cite
If you use our work and are inspired by our work, please consider cite us (available soon):
```bibtex
@article{wang2024charxiv,
  title={CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs},
  author={Wang, Zirui and Xia, Mengzhou and He, Luxi and Chen, Howard and Liu, Yitao and Zhu, Richard and Liang, Kaiqu and Wu, Xindi and Liu, Haotian and Malladi, Sadhika and Chevalier, Alexis and Arora, Sanjeev and Chen, Danqi},
  journal={arXiv preprint arXiv:2406.18521},
  year={2024}
}
```
## 🙌 Contributors and Acknowledgement
**📊 [CharXiv](https://charxiv.github.io/) is built by a team consisting of:**  
[Zirui Wang](https://zwcolin.github.io/), [Mengzhou Xia](https://xiamengzhou.github.io/), [Luxi He](https://twitter.com/luxihelucy), [Howard Chen](https://howard50b.github.io/), [Yitao Liu](https://yitaoliu17.com/), [Richard Zhu](https://richardzhu123.github.io/), [Kaiqu Liang](https://kaiquliang.github.io/), [Xindi Wu](https://xindiwu.github.io/), [Haotian Liu](https://hliu.cc/), [Sadhika Malladi](https://www.cs.princeton.edu/~smalladi/), [Alexis Chevalier](https://pli.princeton.edu/people/alexis-chevalier), [Sanjeev Arora](https://www.cs.princeton.edu/~arora/), [Danqi Chen](https://www.cs.princeton.edu/~danqic/)

[Princeton Language and Intelligence, Princeton University](https://pli.princeton.edu/)  
[University of Wisconsin, Madison](https://www.wisc.edu/)  
[The University of Hong Kong](https://www.hku.hk/)

🤗 We adapted part of the [MathVista](https://github.com/lupantech/MathVista)'s codebase in building our evaluation framework, and we greatly appreciate their contributions to the MLLM community.  
🤗 The lyrics in the teaser video are created by [GPT-4o](https://openai.com/index/hello-gpt-4o/) from our abstract, and the music is created by [Suno](https://suno.com/). Video is manually edited using [CapCut](https://www.capcut.com/).
