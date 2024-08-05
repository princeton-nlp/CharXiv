# Adapted from https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5
# This has support for MiniCPM V2 and V2.5

from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from PIL import Image
import torch

def generate_response(queries, model_path):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(device='cuda', dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    for k in tqdm(queries):
        query = queries[k]['question']
        image = Image.open(queries[k]["figure_path"]).convert('RGB')
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
        queries[k]['response'] = res
