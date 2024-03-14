import json
import os
import pandas as pd

import inflect

from utils import LLAMA_SYS_PROMPT


from utils import * 

stars = '*' * 50


class MixSkillTextGeneration():
    def __init__(self, engine, skills_dict) -> None:
        self.engine = engine
        self.skills_dict = skills_dict

    def get_skill_definition_and_example(self, skill, version="regular"):
        if version == 'regular':
            return f"Skill: {skill}\nDefinition: {self.skills_dict[skill]['definition']}\nExample: {self.skills_dict[skill]['example']}"
        elif version == 'simple':
            return f"**{skill}**: {self.skills_dict[skill]['definition']} For example, {self.skills_dict[skill]['example']}"
        elif version == 'no_example':
            return f"**{skill}**: {self.skills_dict[skill]['definition']}"
        else:
            raise NotImplementedError
    
    def get_prompt(self, list_of_skills, topic, p, num_sentences=-1, prompt_version="-1"):
        skills_str = ', '.join(list_of_skills) 
        
        skills_defs_and_examples = '\n'.join(self.get_skill_definition_and_example(skill) for skill in list_of_skills)
        skills_defs_and_examples_simple = '\n'.join(self.get_skill_definition_and_example(skill, "simple") for skill in list_of_skills)
        skills_defs = '\n'.join(self.get_skill_definition_and_example(skill, "no_example") for skill in list_of_skills)
        
        num_skills = len(list_of_skills)
        num_skills_str = p.number_to_words(num_skills)
        if num_sentences < 0:
            num_sentences = mapping_num_skills_to_num_sentences(num_skills)
        num_sentences_str = p.number_to_words(num_sentences) + ' ' + ('sentences' if num_sentences > 1 else 'sentence')

        
        if prompt_version == "-1":
            prompt_version = "default"
        prompt_file = os.path.join("prompts", "generation", f"{prompt_version}.json")
        if not os.path.exists(prompt_file):
            raise NotImplementedError
        else:
            with open(prompt_file) as f:
                prompts = json.load(f)
            all_prompts = [p.format(
                skills_str=skills_str,
                skills_defs_and_examples=skills_defs_and_examples,
                skills_defs_and_examples_simple=skills_defs_and_examples_simple,
                skills_defs=skills_defs,
                num_skills=num_skills,
                num_skills_str=num_skills_str,
                topic=topic,
                num_sentences=num_sentences,
                num_sentences_str=num_sentences_str,
            ) for p in prompts]

        return all_prompts

    def get_text(self, conv):
        text = conv.messages[-1][1].split('Answer:')[-1]
        text = text.split('Explanation')[0].strip()
        
        return text

    def __call__(self, skills, topic, save_dir=None, prompt_version=-1, idx=-1, **kwargs):
        if save_dir:
            os.makedirs(os.path.join(save_dir, "conv"), exist_ok=True)
            df = load_csv(os.path.join(save_dir, "records.csv"))
            if idx >= 0 and idx < len(df):
                print(f"skipping {idx}, already exists")
                return 
        conv = self.engine.initialize_conversation()
        p = inflect.engine()
        msg = self.get_prompt(skills, topic, p, prompt_version=prompt_version)
        
        all_msgs = []
        result_dict = {
                    'skills': skills,
                    'topic': topic,
                    'system prompt': conv.system_message,
                    'conv': [], 
                    }
        record = {'k': len(skills), 'skills': ', \n'.join(skills), 'topic': topic, 'system prompt': conv.system_message}
        for i, _msg in enumerate(msg):
            conv.append_message(conv.roles[0], _msg)
            conv.append_message(conv.roles[1], None)

            outputs, prompt = self.engine.query(conv, **kwargs) 

            conv.update_last_message(outputs)

            text = self.get_text(conv.copy())
            
            role1, role2 = conv.roles[0], conv.roles[1]
            
            result_dict['conv'].append({f"{role1}": f"{_msg}", f"{role2}": f"{outputs}"})
            
            record[f"{role1}_{i}"] = _msg
            record[f"{role2}_{i}"] = outputs
            record[f"model_input_{i}"] = prompt
            record[f"text_{i}"] = text
            printing = f"\n{stars}\n{role1}: {_msg}\n{role2}: {outputs}\n{stars}\n{text}\n{stars}\n" 
            all_msgs.append(printing)
            
            print(printing, flush=True)

        df.append(record)
        if save_dir is not None:
            print(f'writing to {save_dir}')
            
            topic_str = topic.replace(' ', '_')
            skills_str = '_'.join(skills).replace(' ', '_')
            file_name = os.path.join(save_dir, "conv", f"k_{len(skills)}-topic_{topic_str}-skills_{skills_str}")

            json_object = json.dumps(result_dict, indent=4)
            with open(f"{file_name}.json", "w") as outfile:
                outfile.write(json_object)
                
            all_msgs = '\n'.join(all_msgs)
            with open(f"{file_name}.txt", 'w') as text_file:
                text_file.write(all_msgs)

            df = pd.DataFrame(df)
            df.to_csv(os.path.join(save_dir, "records.csv"), index=False)
            

class MixSkillGrading():
    def __init__(self, engine, skills_dict=None) -> None:
        self.engine = engine
        self.skills_dict = skills_dict
        self.prompt_versions_with_rubric = ["5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "gpt", "llama"]
        
    def get_skill_definition_and_example(self, skill, version="regular"):
        if version == 'regular':
            return f"Skill: {skill}\nDefinition: {self.skills_dict[skill]['definition']}\nExample: {self.skills_dict[skill]['example']}"
        elif version == 'simple':
            return f"**{skill}**: {self.skills_dict[skill]['definition']} For example, {self.skills_dict[skill]['example']}"
        elif version == 'no_example':
            return f"**{skill}**: {self.skills_dict[skill]['definition']}"
        else:
            raise NotImplementedError
    
    def get_prompt(self, skills_str, topic, student_answer, p, num_sentences=-1, prompt_version="-1"):
        if num_sentences==-1:
            num_sentences = mapping_num_skills_to_num_sentences(len(skills_str.split(',')))
        num_sentences_str = p.number_to_words(num_sentences) + ' ' + ('sentences' if num_sentences > 1 else 'sentence')
        
        list_of_skills = skills_str.split(',')
        list_of_skills = [skill.strip() for skill in list_of_skills]
        skills_str = ', '.join(list_of_skills)
        skills_defs_and_examples = '\n'.join(self.get_skill_definition_and_example(skill) for skill in list_of_skills)
        skills_defs_and_examples_simple = '\n'.join(self.get_skill_definition_and_example(skill, "simple") for skill in list_of_skills)
        skills_defs = '\n'.join(self.get_skill_definition_and_example(skill, "no_example") for skill in list_of_skills)

        
        num_skills = len(list_of_skills)
        num_skills_str = p.number_to_words(num_skills)
        num_sentences_str = p.number_to_words(num_sentences) + ' ' + ('sentences' if num_sentences > 1 else 'sentence')

        
        if prompt_version == "-1":
            prompt_version = "gpt"
        prompt_file = os.path.join("prompts", "grade", f"{prompt_version}.json")
        if not os.path.exists(prompt_file):
            raise NotImplementedError
        else:
            rubric_items = ''
            if prompt_version in self.prompt_versions_with_rubric:
                # rubric_start = "Illustrates" if prompt_version in ["8", "9"] else "Correctly illustrates"
                rubric_start = "Correctly illustrates"
                if prompt_version in ["8", "9"]:
                    rubric_start = "Illustrates"
                elif prompt_version in ["13", "14", "llama"]:
                    rubric_start = "Contains"
                
                skills_rubric_items = [f"{rubric_start} {skill.strip().lower()}" for skill in list_of_skills]
                rubric_items = skills_rubric_items + [f"Pertains to {topic}", "Text makes sense", f"At most {num_sentences_str}"]  #[f"Relevant to topic of {topic}", "Makes sense", f"At most {num_sentences_str}"] Text makes sense
                rubric_items = ', '.join(rubric_items)
                
            with open(prompt_file) as f:
                prompts = json.load(f)
                all_prompts = [p.format(
                    skills_str=skills_str,
                    skills_defs_and_examples=skills_defs_and_examples,
                    skills_defs_and_examples_simple=skills_defs_and_examples_simple,
                    skills_defs=skills_defs,
                    student_answer=student_answer,
                    num_skills=num_skills,
                    num_skills_str=num_skills_str,
                    topic=topic,
                    num_sentences=num_sentences,
                    num_sentences_str=num_sentences_str,
                    rubric_items=rubric_items,
                ) for p in prompts]
                
        return all_prompts
       

    def extract_points_manual(self, output):
        return manually_extract_points(output)
    
    def get_score(self, output):
        score = output.lower().split('grade:')[-1].strip()
        while (len(score) > 0) and (not score[0].isdigit()):
            score = score[1:]
        if len(score) == 0:
            return 0.0
        res = ''
        for c in score:
            if c.isdigit() or c=='.':
                res += c
            else:
                break
        while len(res) > 0 and res.endswith('.'):
            res = res[:-1]
        return float(res)

    def get_score_alt(self, output):
        output = output.split("Here's the grading table:")[-1].strip()
        output = output.split("Explanation:")[0].strip()
        
        try:
            output_dict = create_output_dict(output)
        except Exception as e:
            print(f"Error: {e}, output: {output}")
            raise e

        if 'score' not in output_dict.keys():
            output_dict['score'] = 0.0

        return output_dict['score'], output_dict['extracted_score'], output_dict

    def __call__(self, skills, topic, student_answer, idx=-1, nruns=1, save_dir=None, temperature=0.7, repetition_penalty=1.0, max_new_tokens=512, prompt_version=-1, with_system_prompt=False):
        if save_dir:
            save_file = os.path.join(save_dir, "records.csv")
            for i in range(nruns):
                os.makedirs(save_dir.format(r=i+1), exist_ok=True)
            df = [load_csv(save_file.format(r=i+1)) for i in range(nruns)]
            min_len = min([len(dfi) for dfi in df])
            df = [dfi[:min_len] for dfi in df]
            if idx >= 0 and idx < min_len:
                print(f"skipping {idx}, already exists")
                return
        conv = self.engine.initialize_conversation()
        if with_system_prompt:
            conv.system_message = LLAMA_SYS_PROMPT
        p = inflect.engine()
        
        msg = self.get_prompt(skills, topic, student_answer, p, prompt_version=prompt_version)
        assert(len(msg) == 1)
        
        record = {'k': len(skills.split(',')), 'skills': skills, 'topic': topic, 'system prompt': conv.system_message}
        
        num_sentences = mapping_num_skills_to_num_sentences(len(skills.split(',')))
        num_sentences_str = p.number_to_words(num_sentences) + ' ' + ('sentences' if num_sentences > 1 else 'sentence')
        
        for i, _msg in enumerate(msg):
            conv.append_message(conv.roles[0], _msg)
            conv.append_message(conv.roles[1], None)

            outputs_list, prompt = self.engine.query(conv, temperature, repetition_penalty, max_new_tokens, nruns=nruns)

            # check if outputs is a string
            if isinstance(outputs_list, str):
                assert(nruns==1)
                outputs_list = [outputs_list]
            
            # print(outputs_list, flush=True)
            for r, outputs in enumerate(outputs_list):

                conv.update_last_message(outputs)
                
                if prompt_version in self.prompt_versions_with_rubric and prompt_version not in ['14', 'llama']: #=='5':
                    score, score_extracted, output_dict = self.get_score_alt(outputs)
                    total_possible_points = len(skills.split(',')) + 3 # + 3 comes from (topic, clarity, and length req)
                else:
                    score = self.get_score(outputs)
                    points = self.extract_points_manual(outputs)
                    score_extracted = sum(points)
                num_sentences_manual_in_student_answer = count_num_sentences(student_answer)

                role1, role2 = conv.roles[0], conv.roles[1]
                
                record[f"{role1}_{i}"] = _msg
                record[f"{role2}_{i}"] = outputs
                record[f"model_input_{i}"] = prompt
                record[f"score_{i}"] = score
                record[f"score_extracted_{i}"] = score_extracted
                record[f"points_{i}"] = ",".join([str(p) for p in points]) if prompt_version not in self.prompt_versions_with_rubric or prompt_version in ['14', 'llama'] else ",".join(str(val) for key, val in output_dict.items() if 'score' not in key)
                record[f"num_sentences_manual_in_student_answer_{i}"] = num_sentences_manual_in_student_answer
                true_sentence_lim_pt = num_sentences_manual_in_student_answer <= num_sentences
                record[f"true_sentence_lim_pt_{i}"] = float(true_sentence_lim_pt)

                if prompt_version in self.prompt_versions_with_rubric and prompt_version not in ['14', 'llama']: #'5':
                    
                    try:
                        pt_for_sentence_lim = output_dict[f'at most {num_sentences_str}']
                    except Exception as e:
                        for key in output_dict.keys():
                            if f'at most {num_sentences_str}' in key.lower():
                                pt_for_sentence_lim = output_dict[key]
                                break
                        pt_for_sentence_lim=0.0
                    
                    num_sentences_extracted_eq_num_sentences_model = true_sentence_lim_pt==pt_for_sentence_lim
                    record[f"num_sentences_extracted_eq_num_sentences_model_{i}"] = num_sentences_extracted_eq_num_sentences_model
                    
                    printing = f"\n{stars}\n{role1}: {_msg}\n{role2}: {outputs}\n{stars}\n{score=}\n{score_extracted=}\n{total_possible_points=}\n\n{num_sentences_manual_in_student_answer=}\n{num_sentences_extracted_eq_num_sentences_model=}\n{stars}\n" 
                else:
                    pt_for_sentence_lim = points[-1] if len(points) > 0 else 0.0
                    num_sentences_extracted_eq_num_sentences_model = true_sentence_lim_pt==pt_for_sentence_lim
                    record[f"num_sentences_extracted_eq_num_sentences_model_{i}"] = num_sentences_extracted_eq_num_sentences_model
                    printing = f"\n{stars}\n{role1}: {_msg}\n{role2}: {outputs}\n{stars}\n{score=}\n{score_extracted=}\n{points=}\n\n{num_sentences_manual_in_student_answer=}\n{num_sentences_extracted_eq_num_sentences_model=}\n{stars}\n" 
                
                print(printing, flush=True)

                df[r].append(record)

        if save_dir is not None:
            for r in range(nruns):
                save_file = os.path.join(save_dir, "records.csv")
                save_file = save_file.format(r=r+1)
                print(f'writing to {save_file}')

                dfr = pd.DataFrame(df[r])
                dfr.to_csv(save_file, index=False)
            
