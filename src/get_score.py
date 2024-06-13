import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input/output
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--score_prefix', type=str, default='scores-')
    args = parser.parse_args()

    file_path = f"results/{args.score_prefix}{args.model_name}-{args.mode}_{args.split}.json"
    data = json.load(open(file_path))
    scores = [d['score'] for d in data.values()]
    scores = format(100 * sum(scores)/len(scores), '.2f')
    print(f"Split: {args.split}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_name}")
    print(f"Score: {scores}")
