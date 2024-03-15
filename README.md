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

<details open>
  <summary><b>Table of Contents</b></summary>

1. [Usage](#usage)
2. [Contributing a model](#contributing-a-model)

</details>

## Usage

Preparation:

- set `OPENAI_API_KEY` in your environment

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

The output will be saved to `${root}/k${k}_s${seed}_v${prompt_version}_r${generation_round_idx}/${student_model_name}/record.csv`

- **Grading**

```
python grade.py --student_model_name ${student_model_name} \
    --grader_model_name ${grader_model_name} \
    --exp_path ${root}/k${k}_s${seed}_v${generation_prompt_version}_r${generation_round_idx} \
    --suffix ${grading_round_idx}_${grading_prompt_version} \
    --prompt_version ${grading_prompt_version} 
```

The output will be saved to `${root}/k${k}_s${seed}_v${prompt_version}_r${generation_round_idx}/${student_model_name}/graded/by_{grader_model_name}_on_text_1_${grading_round_idx}_${grading_prompt_version}/record.csv`.
    
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
    
Add `--filter_explicit_name` to deduct points for explicitely mentioning the name of skills

Add `--filter_skills {skill_to_filter_seperated_by_comma}` to filter out combinations containing some specific skills

---

## Contributing a Model

Please follow the steps below to add your model to the SKILLMIX leaderboard.

#### 1. Download `requirements.txt`

```bash
pip install -r requirements.txt
```

#### 2. Add Your Model Configuration to `engine.py`

Please add your model's configuration to `engine.py`, and make sure this is still compatible with `test_pipeline.sh`.

#### 3. Update `test_pipeline.sh`

Please update `test_pipeline.sh` with the appropriate arguments for your model.
We encourage you to experiment with various settings for the arguments in `test_pipeline.sh`. To ensure your model performs well, experiment with various generation prompts in `prompts/generation` (and feel free to add your own). 

#### 4. Submit a Pull Request (PR) with Your Changes

Once you are satisfied with your model's performance on the released list of skills and topics, submit a Pull Request to the main project repository with your changes. The PR should include:

- The updated `engine.py` file with your model configuration.
- Any additional generation prompts in `prompts/generation`
- An updated version of `test_pipeline.sh`

#### 5. Fill Out the Linked Form

After submitting your PR, please fill out [this form](https://forms.gle/MrQNNaFnJBbwMDku7). The form asks for:

- Access to your model weights. Please provide a link or a method for us (e.g., the model card on Huggingface) to access the model weights securely.
- An `OpenAI` key (for evaluation). We encourage you to create a single project key, so that you may cancel it after we are done with evaluation.

---

After completing these steps, we will review your request and update you once your model has been added to the leaderboard. Thank you for contributing!

---

## Citation
```
@article{yu2023skillmix,
      title={Skill-Mix: a Flexible and Expandable Family of Evaluations for AI models}, 
      author={Yu, Dingli and Kaur, Simran and Gupta, Arushi and Brown-Cohen, Jonah and Goyal, Anirudh and Arora, Sanjeev},
      journal={arXiv preprint arXiv:2310.17567},
      year={2023}
}
```
