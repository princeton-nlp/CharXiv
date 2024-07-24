import time
from tqdm import tqdm

def generate_response_remote_wrapper(generate_fn, 
        queries, model_path, api_key, client, init_sleep=1, 
        max_retries=10, sleep_factor=1.6):
    for k in tqdm(queries):
        sleep_time = init_sleep
        query = queries[k]['question']
        image = queries[k]["figure_path"]
        curr_retries = 0
        result = None
        while curr_retries < max_retries and result is None:
            try:
                result = generate_fn(image, query, model_path, 
                    api_key=api_key, client=client, random_baseline=False)
            except:
                print(f"Error {curr_retries}, sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
                curr_retries += 1
                sleep_time *= sleep_factor
        if result is None:
            result = "Error in generating response."
            print(f"Error in generating response for {k}")
        queries[k]['response'] = result

def get_client_fn(model_path):
    if model_path in ['claude-3-sonnet-20240229', 
                      'claude-3-opus-20240229', 
                      'claude-3-haiku-20240307', 
                      'claude-3-5-sonnet-20240620']:
        from .claude import get_client_model
    # gemini
    elif model_path in ['gemini-1.5-pro-001', 
                        'gemini-1.0-pro-vision-001', 
                        'gemini-1.5-flash-001']:
        from .gemini import get_client_model
    # gpt
    elif model_path in ['gpt-4o-2024-05-13', 
                        'gpt-4-turbo-2024-04-09', 
                        'gpt-4o-mini-2024-07-18']:
        from .gpt import get_client_model
    # reka
    elif model_path in ['reka-core-20240415', 
                        'reka-flash-20240226', 
                        'reka-core-20240415']:
        from .reka import get_client_model
    # qwen
    elif model_path in ['qwen-vl-max', 
                        'qwen-vl-plus']:
        from .qwen import get_client_model
    else:
        raise ValueError(f"Model {model_path} not supported")
    return get_client_model

def get_generate_fn(model_path):
    model_name = model_path.split('/')[-1]
    # cambrian
    if model_name in ['cambrian-34b']:
        from .cambrian import generate_response
    # chartgemma
    elif model_name in ['chartgemma']:
        from .chartgemma import generate_response
    # claude
    elif model_name in ['claude-3-sonnet-20240229',
                        'claude-3-opus-20240229',
                        'claude-3-haiku-20240307',
                        'claude-3-5-sonnet-20240620']:
        from .claude import generate_response
    # deepseekvl
    elif model_name in ['deepseek-vl-7b-chat']:
        from .deepseekvl import generate_response
    # gemini
    elif model_name in ['gemini-1.5-pro-001', 
                        'gemini-1.0-pro-vision-001', 
                        'gemini-1.5-flash-001']:
        from .gemini import generate_response
    # gpt
    elif model_name in ['gpt-4o-2024-05-13', 
                        'gpt-4-turbo-2024-04-09', 
                        'gpt-4o-mini-2024-07-18']:
        from .gpt import generate_response
    # idefics2
    elif model_name in ['idefics2-8b',
                        'idefics2-8b-chatty']:
        from .idefics2 import generate_response
    # ixc2
    elif model_name in ['internlm-xcomposer2-4khd-7b',
                        'internlm-xcomposer2-vl-7b']:
        from .ixc2 import generate_response
    # internvl2
    elif model_name in ['InternVL2-26B',
                        'InternVL2-Llama3-76B']:
        from .internvl2 import generate_response
    # internvl15
    elif model_name in ['InternVL-Chat-V1-5']:
        from .internvl15 import generate_response
    # llava16
    elif model_name in ['llava-v1.6-34b-hf',
                        'llava-v1.6-mistral-7b-hf']:
        from .llava16 import generate_response
    # mgm
    elif model_name in ['MGM-34B-HD',
                        'MGM-8B-HD']:
        from .mgm import generate_response
    # minicpm
    elif model_name in ['MiniCPM-Llama3-V-2_5',
                        'MiniCPM-V-2']:
        from .minicpm import generate_response
    # moai
    elif model_name in ['MoAI-7B']:
        from .moai import generate_response
    # paligemma
    elif model_name in ['paligemma-3b-mix-448']:
        from .paligemma import generate_response
    # phi3
    elif model_name in ['Phi-3-vision-128k-instruct']:
        from .phi3 import generate_response
    # qwen
    elif model_name in ['qwen-vl-max',
                        'qwen-vl-plus']:
        from .qwen import generate_response
    # reka
    elif model_name in ['reka-core-20240415',
                        'reka-flash-20240226',
                        'reka-core-20240415']:
        from .reka import generate_response
    # sphinx
    elif model_name in ['SPHINX-v2-1k']:
        from .sphinx2 import generate_response
    # vila
    elif model_name in ['VILA1.5-40b']:
        from .vila15 import generate_response
    else:
        raise ValueError(f"Model {model_name} not supported")
    return generate_response
