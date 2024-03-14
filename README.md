# SKILL-MIX: a Flexible and Expandable Family of Evaluations for AI models  

## What is this repo?

- This repo is a reference implementation of SKILL-MIX evaluation, as described in this paper [Skill-Mix: a Flexible and Expandable Family of Evaluations for AI models](https://arxiv.org/abs/2310.17567). SKILL-MIX evaluates the capability of Large Language Models (LLMs) to combine language skills. It contains a set of 101 language skills (e.g., “use of metaphor”) and a set of 100 topics (e.g., “dueling”, “gardening”). Every time it **randomly** picks **$k$ language skills** from the skill set and **one topic** from the topic set, and asks LLMs to generate **a short piece of text in the context of the selected topic and illustrate all $k$ selected skills**. Here is an example:
  
<p align="center">
 <img width="500" alt="image" src="https://github.com/LeoYu/skill-mix/assets/1962796/aff46d3d-e18a-4c4d-933a-532297fb9be0">
 </p>
  
  The generations by LLMs are then graded by GPT-4 or LLaMA-2 70B chat.
  Below is the full pipeline of SKILL-MIX 
   
<p align="center">
 <img width="1000" alt="image" src="https://github.com/LeoYu/skill-mix/assets/1962796/f1ae0bb2-63f7-4b3e-ab19-0e6ba4e5e70b">
 </p>
 
- Since the goal of SKILL-MIX is to test general-purpose text generation capability rather than ability on the particular set of skills and topics, we release only a random subset of 10% of [skills](skills.csv) and [topics](topics.txt) in this repo. 
- We provide instructions to submit and test your model in the full sets of skills of topics. 

## usage

Preparation:

- set OPENAI_API_KEY in your environment

Parameters:

- `k`: number of skills to combine
- `student_model_name`: the short model name for generation model (Student), options are:
  *  `lama2-7b-chat`, `llama2-13b-chat`, `llama2-70b-chat`
  *  `gpt-4`, `gpt-3.5-turbo`
  *  `mistral-7b-chat`, `qwen-14b-chat`, `xwin-lm-70b`, `falcon-180b-chat`, `tigerbot-70b-chat`
- `grader_model_name`: the short name for grading model (Grader)
- `num_gpus_for_student`: number of gpus to use for generation.
- `num_gpus_for_grader`: number of gpus to use for grading.
- `root`: root path of the experiments
- `num_combinations`: number of skill and topic combinations to generation
- `seed`: seed for sampling the combinations
- `generation_prompt_version`: the prompt used for generation, stored in `prompts/generation/{generation_prompt_version}.json`
- `grading_prompt_version`: the prompt used for grading, stored in `prompts/grading/{grading_prompt_version}.json`
- `num_generation_rounds`: number of rounds for generation
- `generation_round_idx`: the index of the current generation round, select from `{1, ..., num_generation_rounds}`
- `num_grading_rounds`: number of rounds for generation
- `grading_round_idx`: the index of the current generation round, select from `{1, ..., num_grading_rounds}`
- `skill_csv_file`: csv file for skill definition and example
- `topic_txt_file`: text file for the list of topics


Commands:

- **Query random combination**
```
python query_random.py --k ${k} \
    --generation_round_idx ${generation_round_idx} \
    --model_name ${student_model_name} --num_gpus ${num_gpus_for_student} \
    --exp_path ${root} \
    --num_combinations ${num_combinations} --seed ${seed} \
    --prompt_version ${generation_prompt_version} \
    --skills_dict_path ${skill_csv_file} --topics_path ${topic_txt_file}
```

Output will be saved to `${root}/k${k}_s${seed}_v${prompt_version}_r${generation_round_idx}/${student_model_name}/record.csv`

- **Grading**

```
python grade.py --student_model_name ${student_model_name} \
    --grader_model_name ${grader_model_name} \
    --exp_path ${root}/k${k}_s${seed}_v${generation_prompt_version}_r${generation_round_idx} \
    --suffix ${grading_round_idx}_${grading_prompt_version} \
    --prompt_version ${grading_prompt_version} 
```

Output will be saved to `${root}/k${k}_s${seed}_v${prompt_version}_r${generation_round_idx}/${student_model_name}/graded/by_{grader_model_name}_on_text_1_${grading_round_idx}_${grading_prompt_version}/record.csv`.
    
For GPT-4, we allow multiple runs of grading in the same time 
```
python grade.py --student_model_name ${student_model_name} \
    --grader_model_name ${grader_model_name} --num_gpus ${num_gpus_for_grader} \
    --exp_path ${root}/k${k}_s${seed}_v${generation_prompt_version}_r${generation_round_idx} \
    --suffix {r}_${grading_prompt_version} --nruns ${num_generation_rounds} \
    --prompt_version ${grading_prompt_version} 
```

For grading prompt version `gpt`, you can add `--max-new-tokens $((120+20*k))` for just obtain the scores.

- **Aggregate Metrics**

```
python aggregate_metrics.py --student_model_name ${student_model_name} --grader_model_name ${grader_model_name} \
    --exp_path ${root} --k_range ${list_of_k_separated_by_comma} \
    --num_generation_rounds ${num_generation_rounds} --num_generations_per_round ${num_combinations} --num_grading_rounds ${num_grading_rounds} \
    --generation_prompt_version ${generation_prompt_version} --grading_prompt_version ${grading_prompt_version} --seed ${seed}
```
    
add `--filter_explicit_name` to deduct points for explicitely mentioning the name of skills

add `--filter_skills {skill_to_filter_seperated_by_comma}` to filter out combinations containing some specific skills

## Citation
```
@article{yu2023skillmix,
      title={Skill-Mix: a Flexible and Expandable Family of Evaluations for AI models}, 
      author={Yu, Dingli and Kaur, Simran and Gupta, Arushi and Brown-Cohen, Jonah and Goyal, Anirudh and Arora, Sanjeev},
      journal={arXiv preprint arXiv:2310.17567},
      year={2023}
}
```
