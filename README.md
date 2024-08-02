# 📊 CharXiv
🏠[Home](https://charxiv.github.io/) | 🤗[Data](https://huggingface.co/datasets/princeton-nlp/CharXiv) | 🥇[Leaderboard](https://charxiv.github.io/#leaderboard) | 📄[Paper](https://arxiv.org/abs/2406.18521) | *Current Version: v1.0*

This repository contains the code to evaluate models on CharXiv from the paper [CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs](https://arxiv.org/abs/2406.18521).

*🤗 We are first-time evaluation suite builders and this codebase is released for the first-time. We are committed to improving it. If you have any questions, feel free to raise issues and/or submit pull requests for new features of bug fixes.*

### 🔊 An 80-second video teaser of CharXiv

https://github.com/princeton-nlp/CharXiv/assets/59942464/ab9b293b-8fd6-4735-b8b3-0079ee978b61

## 📰 News
**[07/26/2024]** 🚀 Upcoming this week: we'll be releasing scores for [GPT-4o-mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) as well as the largest and most capable open-weight VLM in our benchmark: [InternVL2 LLaMA-3 76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B). Alongside scores, we find some [interesting patterns](https://x.com/zwcolin/status/1816948825036071196) in the trend of model improvement with respect to differnet chart understanding benchmarks on X.
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

## 🏆 Leaderboard

<table><tbody><tr><th>Metadata</th><th></th><th></th><th>Reasoning</th><th></th><th></th><th></th><th></th><th>Descriptive</th><th></th><th></th><th></th><th></th><th></th></tr><tr><td>Model</td><td>Weight</td><td>Size [V/L] (B)</td><td>Overall</td><td>TC</td><td>TG</td><td>NC</td><td>NG</td><td>Overall</td><td>INEX</td><td>ENUM</td><td>PATT</td><td>CNTG</td><td>COMP</td></tr><tr><td>Human</td><td>N/A</td><td>Unknown</td><td>80.50</td><td>77.27</td><td>77.78</td><td>84.91</td><td>83.41</td><td>92.10</td><td>91.40</td><td>91.20</td><td>95.63</td><td>93.38</td><td>92.86</td></tr><tr><td>Claude 3.5 Sonnet</td><td>Proprietary</td><td>Unknown</td><td>60.20</td><td>61.14</td><td>78.79</td><td>63.79</td><td>46.72</td><td>84.30</td><td>82.62</td><td>88.86</td><td>90.61</td><td>90.08</td><td>48.66</td></tr><tr><td>GPT-4o</td><td>Proprietary</td><td>Unknown</td><td>47.10</td><td>50.00</td><td>61.62</td><td>47.84</td><td>34.50</td><td>84.45</td><td>82.44</td><td>89.18</td><td>90.17</td><td>85.50</td><td>59.82</td></tr><tr><td>Gemini 1.5 Pro</td><td>Proprietary</td><td>Unknown</td><td>43.30</td><td>45.68</td><td>56.57</td><td>45.69</td><td>30.57</td><td>71.97</td><td>81.79</td><td>64.73</td><td>79.48</td><td>76.34</td><td>15.18</td></tr><tr><td>InternVL Chat V2.0 Pro</td><td>Proprietary</td><td>Unknown</td><td>39.80</td><td>40.00</td><td>60.61</td><td>44.40</td><td>25.76</td><td>76.83</td><td>77.11</td><td>84.67</td><td>77.07</td><td>78.88</td><td>27.23</td></tr><tr><td>InternVL Chat V2.0 76B</td><td>Open</td><td>5.9 / 70</td><td>38.90</td><td>40.00</td><td>59.60</td><td>42.67</td><td>24.02</td><td>75.17</td><td>77.11</td><td>78.69</td><td>76.20</td><td>79.13</td><td>32.14</td></tr><tr><td>GPT-4V</td><td>Proprietary</td><td>Unknown</td><td>37.10</td><td>38.18</td><td>57.58</td><td>37.93</td><td>25.33</td><td>79.92</td><td>78.29</td><td>85.79</td><td>88.21</td><td>80.92</td><td>41.07</td></tr><tr><td>GPT-4o Mini</td><td>Proprietary</td><td>Unknown</td><td>34.10</td><td>35.23</td><td>47.47</td><td>32.33</td><td>27.95</td><td>74.92</td><td>74.91</td><td>82.81</td><td>69.21</td><td>79.13</td><td>35.71</td></tr><tr><td>Gemini 1.5 Flash</td><td>Proprietary</td><td>Unknown</td><td>33.90</td><td>36.36</td><td>54.55</td><td>30.60</td><td>23.58</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>InternVL Chat V2.0 26B</td><td>Open</td><td>5.9 / 20</td><td>33.40</td><td>33.18</td><td>51.52</td><td>41.81</td><td>17.47</td><td>62.40</td><td>71.35</td><td>61.02</td><td>55.90</td><td>67.94</td><td>6.25</td></tr><tr><td>Claude 3 Sonnet</td><td>Proprietary</td><td>Unknown</td><td>32.20</td><td>31.59</td><td>50.51</td><td>31.47</td><td>26.20</td><td>73.65</td><td>75.74</td><td>81.92</td><td>76.64</td><td>72.26</td><td>8.48</td></tr><tr><td>Claude 3 Haiku</td><td>Proprietary</td><td>Unknown</td><td>31.80</td><td>29.77</td><td>45.45</td><td>34.48</td><td>27.07</td><td>65.08</td><td>69.87</td><td>69.98</td><td>64.85</td><td>61.83</td><td>8.04</td></tr><tr><td>Phi-3 Vision</td><td>Open</td><td>0.3 / 4</td><td>31.60</td><td>31.36</td><td>46.46</td><td>35.78</td><td>21.40</td><td>60.48</td><td>67.62</td><td>61.18</td><td>54.59</td><td>65.39</td><td>6.25</td></tr><tr><td>Claude 3 Opus</td><td>Proprietary</td><td>Unknown</td><td>30.20</td><td>26.36</td><td>50.51</td><td>33.62</td><td>25.33</td><td>71.55</td><td>75.62</td><td>73.69</td><td>73.58</td><td>70.48</td><td>26.79</td></tr><tr><td>InternVL Chat V1.5</td><td>Open</td><td>5.9 / 20</td><td>29.20</td><td>30.00</td><td>45.45</td><td>32.33</td><td>17.47</td><td>58.50</td><td>69.63</td><td>52.95</td><td>53.06</td><td>64.63</td><td>5.80</td></tr><tr><td>Reka Core</td><td>Proprietary</td><td>Unknown</td><td>28.90</td><td>27.50</td><td>41.41</td><td>28.45</td><td>26.64</td><td>55.60</td><td>58.90</td><td>50.52</td><td>65.72</td><td>71.25</td><td>10.71</td></tr><tr><td>Ovis 1.5 Gemma2 9B</td><td>Open</td><td>0.4 / 9</td><td>28.40</td><td>26.14</td><td>44.44</td><td>33.19</td><td>20.96</td><td>62.60</td><td>64.29</td><td>71.75</td><td>56.33</td><td>66.16</td><td>5.80</td></tr><tr><td>Ovis 1.5 Llama3 8B</td><td>Open</td><td>0.4 / 8</td><td>28.20</td><td>27.27</td><td>49.49</td><td>31.03</td><td>17.90</td><td>60.15</td><td>61.39</td><td>68.93</td><td>56.33</td><td>61.83</td><td>7.14</td></tr><tr><td>Cambrian 34B</td><td>Open</td><td>1.9 / 34</td><td>27.30</td><td>24.55</td><td>44.44</td><td>27.59</td><td>24.89</td><td>59.73</td><td>59.31</td><td>70.94</td><td>53.28</td><td>64.63</td><td>5.36</td></tr><tr><td>Reka Flash</td><td>Proprietary</td><td>Unknown</td><td>26.60</td><td>26.59</td><td>39.39</td><td>30.60</td><td>17.03</td><td>56.45</td><td>61.39</td><td>48.59</td><td>69.87</td><td>72.52</td><td>7.14</td></tr><tr><td>Mini Gemini HD Yi 34B</td><td>Open</td><td>0.5 / 34</td><td>25.00</td><td>26.59</td><td>43.43</td><td>27.16</td><td>11.79</td><td>52.68</td><td>53.86</td><td>55.04</td><td>65.50</td><td>53.94</td><td>2.23</td></tr><tr><td>InternLM XComposer2 4KHD</td><td>Open</td><td>0.3 / 7</td><td>25.00</td><td>23.86</td><td>43.43</td><td>29.31</td><td>14.85</td><td>54.65</td><td>61.09</td><td>54.08</td><td>51.53</td><td>59.80</td><td>6.70</td></tr><tr><td>MiniCPM-V2.5</td><td>Open</td><td>0.4 / 8</td><td>24.90</td><td>25.23</td><td>43.43</td><td>25.43</td><td>15.72</td><td>59.27</td><td>62.28</td><td>61.90</td><td>56.77</td><td>68.96</td><td>10.27</td></tr><tr><td>Qwen VL Max</td><td>Proprietary</td><td>Unknown</td><td>24.70</td><td>26.14</td><td>41.41</td><td>24.57</td><td>14.85</td><td>41.48</td><td>50.42</td><td>28.41</td><td>53.71</td><td>51.15</td><td>4.46</td></tr><tr><td>VILA 1.5 40B</td><td>Open</td><td>5.9 / 34</td><td>24.00</td><td>21.59</td><td>41.41</td><td>25.00</td><td>20.09</td><td>38.67</td><td>42.88</td><td>29.62</td><td>51.31</td><td>50.89</td><td>9.82</td></tr><tr><td>Reka Edge</td><td>Proprietary</td><td>Unknown</td><td>23.50</td><td>20.23</td><td>32.32</td><td>30.60</td><td>18.78</td><td>33.65</td><td>36.65</td><td>28.49</td><td>34.72</td><td>52.16</td><td>4.91</td></tr><tr><td>Gemini 1.0 Pro</td><td>Proprietary</td><td>Unknown</td><td>22.80</td><td>20.91</td><td>48.48</td><td>18.10</td><td>20.09</td><td>54.37</td><td>67.97</td><td>39.23</td><td>60.48</td><td>62.60</td><td>8.93</td></tr><tr><td>LLaVA 1.6 Yi 34B</td><td>Open</td><td>0.3 / 34</td><td>22.50</td><td>20.45</td><td>37.37</td><td>23.71</td><td>18.78</td><td>51.05</td><td>46.38</td><td>63.44</td><td>56.11</td><td>51.91</td><td>5.80</td></tr><tr><td>Mini Gemini HD Llama3 8B</td><td>Open</td><td>0.5 / 8</td><td>19.00</td><td>19.77</td><td>36.36</td><td>21.12</td><td>7.86</td><td>44.42</td><td>49.41</td><td>39.23</td><td>51.09</td><td>55.98</td><td>1.79</td></tr><tr><td>InternLM XComposer2</td><td>Open</td><td>0.3 / 7</td><td>18.70</td><td>16.14</td><td>38.38</td><td>21.98</td><td>11.79</td><td>38.75</td><td>34.10</td><td>43.58</td><td>46.72</td><td>52.93</td><td>5.80</td></tr><tr><td>MiniCPM-V2</td><td>Open</td><td>0.4 / 2.4</td><td>18.50</td><td>17.95</td><td>33.33</td><td>19.40</td><td>12.23</td><td>35.77</td><td>39.74</td><td>36.56</td><td>26.42</td><td>44.53</td><td>5.36</td></tr><tr><td>IDEFICS 2</td><td>Open</td><td>0.4 / 7</td><td>18.20</td><td>15.45</td><td>35.35</td><td>17.24</td><td>17.03</td><td>32.77</td><td>36.12</td><td>27.28</td><td>40.83</td><td>43.26</td><td>3.12</td></tr><tr><td>IDEFICS 2 Chatty</td><td>Open</td><td>0.4 / 7</td><td>17.80</td><td>15.45</td><td>34.34</td><td>19.83</td><td>13.10</td><td>41.55</td><td>34.88</td><td>54.56</td><td>45.63</td><td>44.27</td><td>6.70</td></tr><tr><td>MoAI</td><td>Open</td><td>0.3 / 7</td><td>17.50</td><td>9.32</td><td>36.36</td><td>21.12</td><td>21.40</td><td>28.70</td><td>31.20</td><td>21.23</td><td>39.96</td><td>40.46</td><td>7.59</td></tr><tr><td>DeepSeek VL</td><td>Open</td><td>0.5 / 7</td><td>17.10</td><td>16.36</td><td>32.32</td><td>19.83</td><td>9.17</td><td>45.80</td><td>49.11</td><td>45.20</td><td>42.79</td><td>60.31</td><td>4.91</td></tr><tr><td>SPHINX V2</td><td>Open</td><td>1.9 / 13</td><td>16.10</td><td>13.86</td><td>28.28</td><td>17.67</td><td>13.54</td><td>30.25</td><td>35.59</td><td>24.37</td><td>41.05</td><td>29.52</td><td>1.79</td></tr><tr><td>Qwen VL Plus</td><td>Proprietary</td><td>Unknown</td><td>16.00</td><td>15.45</td><td>45.45</td><td>12.07</td><td>8.30</td><td>28.93</td><td>33.33</td><td>17.92</td><td>32.10</td><td>56.23</td><td>2.23</td></tr><tr><td>LLaVA 1.6 Mistral 7B</td><td>Open</td><td>0.3 / 7</td><td>13.90</td><td>11.36</td><td>32.32</td><td>16.81</td><td>7.86</td><td>35.40</td><td>34.70</td><td>33.98</td><td>48.91</td><td>42.49</td><td>8.48</td></tr><tr><td>ChartGemma</td><td>Open</td><td>0.4 / 2</td><td>12.50</td><td>11.59</td><td>24.24</td><td>16.81</td><td>4.80</td><td>21.30</td><td>27.58</td><td>18.97</td><td>14.19</td><td>19.59</td><td>4.46</td></tr><tr><td>Random (GPT-4o)</td><td>N/A</td><td>Unknown</td><td>10.80</td><td>4.32</td><td>39.39</td><td>5.60</td><td>16.16</td><td>19.85</td><td>21.65</td><td>16.71</td><td>23.80</td><td>25.70</td><td>5.36</td></tr></tbody></table>

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
