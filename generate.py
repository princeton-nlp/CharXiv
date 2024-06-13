import os, json, argparse
from tqdm import tqdm

# sample code to evaluate the IXC2 4khd model
# https://huggingface.co/internlm/internlm-xcomposer2-4khd-7b
def demo(queries, model_path=None):
    import torch
    from transformers import AutoModel, AutoTokenizer
    assert model_path is not None, "Model path is required for demo"
    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, 
                                      trust_remote_code=True).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    for k in tqdm(queries):
        query = '<ImageHere>' + queries[k]['question']
        image = queries[k]["figure_path"]
        with torch.cuda.amp.autocast():
            response, _ = model.chat(tokenizer, query=query, image=image, 
                                     hd_num=16, history=[], do_sample=False)
        queries[k]['response'] = response

def evaluate(queries):
    """Evaluate the model on the given queries.

    Parameters:
    queries (dict): Dictionary of queries to evaluate. Each query should have the following keys:
        - figure_path (str): Path to the image file
        - question (str): Question to ask about the image
    
    Returns:
    None
    """
    raise NotImplementedError("Implement your own evaluation pipeline based on your model design")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input/output
    parser.add_argument('--data_dir', type=str, required=False, default="./data", \
                        help="Directory containing the input json files")
    parser.add_argument('--image_dir', type=str, required=False, default="./images", \
                        help="Directory containing the images")
    parser.add_argument('--output_dir', type=str, required=False, default="./results", \
                        help="Directory to save the output json files")
    parser.add_argument('--split', type=str, required=False, choices=['val', 'test'], default='val',
                        help="Split of the data")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['descriptive', 'reasoning'],
                        help="Mode of the evaluation")

    # custom arguments
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    input_file = os.path.join(args.data_dir, f"{args.mode}_{args.split}.json")
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)

    # output file
    os.makedirs(args.output_dir, exist_ok=True)
    assert '-' not in args.model_name, "Model name cannot contain '-'"
    output_file = os.path.join(args.output_dir, 
            f'gen-{args.model_name}-{args.mode}_{args.split}.json')

    if args.mode == 'descriptive':
        from descriptive_utils import build_descriptive_quries
        queries = build_descriptive_quries(data, args.image_dir)
    elif args.mode == 'reasoning':
        from reasoning_utils import build_reasoning_queries
        queries = build_reasoning_queries(data, args.image_dir)
    else: 
        raise ValueError("Mode not supported")
    
    print("Number of test problems to run:", len(queries))
    print("Evaluation mode:", args.mode)
    print("Output file:", output_file)

    evaluate(queries) # switch to demo(queries, model_path) for evaluating the IXC2 4khd model

    for k in queries:
        queries[k].pop("figure_path", None)
        queries[k].pop("question", None)

    try:
        print(f"Saving results to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w+") as f:
            json.dump(queries, f, indent=4)
        print(f"Results saved.")
    except Exception as e:
        print(e)
        print(f"Error in saving {output_file}")
