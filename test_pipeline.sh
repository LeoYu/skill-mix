export OPENAI_API_KEY=xx-xxxxxxxxxxxxxxxxxxxxxx

k=3
student_model_name=gpt-4
grader_model_name=gpt-4
num_gpus_for_student=0
num_gpus_for_grader=0
root=test
num_combinations=5
seed=10
generation_prompt_version=default
grading_prompt_version=gpt
num_generation_rounds=3
num_grading_rounds=3
grading_round_idx=1
skill_csv_file=skills.csv
topic_txt_file=topics.txt

for generation_round_idx in $(seq 1 ${num_generation_rounds})
do 
    python query_random.py --k ${k} \
        --generation_round_idx ${generation_round_idx} \
        --model_name ${student_model_name} --num_gpus ${num_gpus_for_student} \
        --exp_path ${root} \
        --num_combinations ${num_combinations} --seed ${seed} \
        --prompt_version ${generation_prompt_version} \
        --skills_dict_path ${skill_csv_file} --topics_path ${topic_txt_file}
    
    python grade.py --student_model_name ${student_model_name} --grader_model_name ${grader_model_name} --num_gpus ${num_gpus_for_grader} \
        --exp_path ${root}/k${k}_s${seed}_v${generation_prompt_version}_r${generation_round_idx} \
        --suffix {r}_${grading_prompt_version} --nruns ${num_grading_rounds} \
        --prompt_version ${grading_prompt_version}
done

python aggregate_metrics.py --student_model_name ${student_model_name} --grader_model_name ${grader_model_name} --exp_path ${root} --k_range $k --num_generation_rounds ${num_generation_rounds} --num_generations_per_round ${num_combinations} --num_grading_rounds ${num_grading_rounds} --generation_prompt_version ${generation_prompt_version} --grading_prompt_version ${grading_prompt_version} --seed ${seed}

