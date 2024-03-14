from itertools import combinations
import argparse
import os
import random

from fastchat.model import add_model_args
from pathlib import Path

from utils import path_to_models_dict, get_skills_dict
from engine import get_engine
from task import MixSkillTextGeneration
from tqdm import tqdm

def get_combinations(elements, k, num=-1):
    if num > 0:
        res = []
        for i in range(num):
            random.shuffle(elements)
            res.append(elements[:k])
        return res
    else:
        return sorted(combinations(elements, k))


def main(args):
    print(args)
    k = args.k
    skills_dict = get_skills_dict(args.skills_dict_path)
    skills = list(skills_dict.keys())
    random.seed(args.seed)
    if args.skill_list_len > 0:
        skills = skills[:args.skill_list_len]
    comb_list = get_combinations(skills, k, args.num_combinations)

    topics = Path(args.topics_path).read_text().split('\n') 
    topics = [topic for topic in topics if len(topic)>0]
    topics_list = []
    if args.fix_topics:
        random.shuffle(topics)
        topics_list = [topics[:args.num_topics]] * len(comb_list)
    else:
        for comb in comb_list:
            random.shuffle(topics)
            topics_list.append(topics[:args.num_topics])
    random.seed(args.exp_path)
    
    model_path = path_to_models_dict.get(args.model_name, args.model_name)
    save_dir = os.path.join(args.exp_path, f'k{k}_s{args.seed}_v{args.prompt_version}_r{args.generation_round_idx}', args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    engine = get_engine(model_path, args)

    text_gen = MixSkillTextGeneration(engine, skills_dict)

    idx = 0
    for comb, topics in tqdm(list(zip(comb_list, topics_list))):
        for topic in topics:
            random.shuffle(comb)
            text_gen(comb, topic, save_dir=save_dir, idx=idx, temperature=args.temperature, repetition_penalty=args.repetition_penalty, max_new_tokens=args.max_new_tokens, prompt_version=args.prompt_version)
        idx += 1

    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--model_name", type=str, default='llama2-13b-chat') # will convert to model path using path_to_models_dict in utils.py, if not found, will use the model_name as the model path 
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--exp_path", type=str, default='test')
    parser.add_argument("--skills_dict_path", type=str, default='skills.csv')
    parser.add_argument("--skill_list_len", type=int, default=-1)
    parser.add_argument("--topics_path", type=str, default="topics.txt")
    parser.add_argument("--num_combinations", type=int, default=3, help="number of combinations of skills to generate, -1 means all possible combinations")
    parser.add_argument("--num_topics", type=int, default=1, help="number of topics to generate for each combination of skills")
    parser.add_argument("--fix_topics", action="store_true", help="fix topics for each combination of skills")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7) # may not be actually used for some engines
    parser.add_argument("--repetition_penalty", type=float, default=1.0) # may not be actually used for some engines
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompt_version", type=str, default="-1")
    parser.add_argument("--generation_round_idx", type=int, default=1)

    args = parser.parse_args()
    main(args)
