import argparse
import os
import random

from fastchat.model import add_model_args

from utils import get_skills_dict, path_to_models_dict 
from engine import get_engine
from task import MixSkillTextGeneration

def main(args):
    print(args)
    skills_dict = get_skills_dict(args.skills_dict_path)
    skills = args.skill_list.split(',')
    skills = [skill.strip() for skill in skills]
    
    model_path = path_to_models_dict.get(args.model_name, args.model_name)
    save_dir = os.path.join(args.exp_path, args.topic + '-' + ','.join([skill.replace(' ', '_') for skill in skills]), args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    engine = get_engine(model_path, args)

    text_gen = MixSkillTextGeneration(engine, skills_dict)

    random.shuffle(skills)
    text_gen(skills, args.topic, save_dir=save_dir, temperature=args.temperature, repetition_penalty=args.repetition_penalty, max_new_tokens=args.max_new_tokens, prompt_version=args.prompt_version)

    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--model_name", type=str, default='llama2-13b-chat')
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--exp_path", type=str, default='on_demand')
    parser.add_argument("--skills_dict_path", type=str, default='skills-9_25.csv')
    parser.add_argument("--skill_list", type=str, default="self serving bias,metaphor,statistical syllogism,folk physics (common knowledge physics)") # "hyperbole, equivocation (informal fallacy), ad hominem, fallacy of division")
    parser.add_argument("--topic", type=str, default="Dueling")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompt_version", type=str, default="-1")

    args = parser.parse_args()
    main(args)
