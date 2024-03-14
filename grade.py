import argparse
import os

from fastchat.model import add_model_args

from utils import path_to_models_dict, get_skills_dict
from engine import get_engine
from task import MixSkillGrading
from tqdm import tqdm
import csv

def extract_between_quotes(text):
    quotation_mark = "\""
    first_idx, last_idx = text.find(quotation_mark), text.rfind(quotation_mark)
    if first_idx!=last_idx:
        return text[first_idx+1:last_idx]
    return text


def main(args):
    print(args)
    save_dir = os.path.join(args.exp_path, args.student_model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = path_to_models_dict.get(args.grader_model_name, args.grader_model_name)
    
    engine = get_engine(model_path, args)

    skills_dict = get_skills_dict(args.skills_dict_path)
    grade = MixSkillGrading(engine, skills_dict)
        
    load_file_path = os.path.join(save_dir, "records.csv")
    delimiter = ','
    
    with open(load_file_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        data = [row for row in reader]
        
    
    row = data[0]
    column_name = args.column_name
    if column_name=='':
        if "text_0" in row:
            column_name = "text" 
        elif "assistant_0" in row:
            column_name = "assistant"
        elif "[/INST]_0" in row:
            column_name = "[/INST]"
        elif "0_0" in row:
            column_name = "0"
    
    column_names = [f"{column_name}_{i}" for i in range(3) if f"{column_name}_{i}" in row]

    key = column_names[args.column_id]

    suffix = '_' + args.suffix if args.suffix else ''
    save_dir = f"{save_dir}/graded/by_{args.grader_model_name}_on_{key}{suffix}"
    # os.makedirs(save_dir, exist_ok=True)

    idx = 0
    for row in tqdm(data):
        skills_str = ', '.join(row['skills'].split(delimiter))
        topic = row['topic']
        
        student_answer = row[key]
        if column_name == '[/INST]':
            student_answer = extract_between_quotes(student_answer)
            student_answer = student_answer.split('\n')[0]

        grade(skills_str, topic, student_answer, idx=idx, nruns=args.nruns, save_dir=save_dir, temperature=args.temperature, repetition_penalty=args.repetition_penalty, max_new_tokens=args.max_new_tokens, prompt_version=args.prompt_version, with_system_prompt=args.with_system_prompt)
        idx += 1


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--grader_model_name", type=str, default='gpt-4')
    parser.add_argument("--student_model_name", type=str, default='llama2-13b-chat')
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--exp_path", type=str, default='test')
    parser.add_argument("--skills_dict_path", type=str, default='skills.csv')
    parser.add_argument("--num_combinations", type=int, default=3, help="number of combinations of skills to generate, -1 means all possible combinations")
    parser.add_argument("--nruns", type=int, default=1, help="number of runs")
    parser.add_argument("--temperature", type=float, default=0.7) # may not be actually used for some engines
    parser.add_argument("--repetition_penalty", type=float, default=1.0) # may not be actually used for some engines
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--column_name", type= str, default="", help="column names of the csv file to grade")
    parser.add_argument("--column_id", type= int, default=-1, help="column id of the csv file to grade")
    parser.add_argument("--prompt_version", type=str, default="-1")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--with_system_prompt", type=bool, default=False)
    

    args = parser.parse_args()
    main(args)
