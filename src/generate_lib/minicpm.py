# Adapted from https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5
# This has support for MiniCPM V2 and V2.5, and V2.6

from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from PIL import Image
import torch

def generate_response(queries, model_path):
    # sdpa attn impl for v2.6, default for 2 and 2.5
    if "MiniCPM-V-2_6" in model_path:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    else:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    for k in tqdm(queries):
        query = queries[k]['question']
        image = Image.open(queries[k]["figure_path"]).convert('RGB')
        if model_path.endswith("MiniCPM-V-2"):
            msgs = [{'role': 'user', 'content': query}]
            res, context, _ = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=False,
                temperature=0.0,
                top_p=1.0,
            )
        # for 2.5
        elif model_path.endswith("MiniCPM-Llama3-V-2_5"):
            msgs = [{'role': 'user', 'content': query}]
            res = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False,
                temperature=0.0,
                top_p=1.0,
            )
        # for 2.6
        elif model_path.endswith("MiniCPM-V-2_6"):
            msgs = [{'role': 'user', 'content': [image, query]}]
            res = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False,
                temperature=0.0,
                top_p=1.0,
            )
        else:
            raise NotImplementedError(f"Model path {model_path} not supported") 
        queries[k]['response'] = res
