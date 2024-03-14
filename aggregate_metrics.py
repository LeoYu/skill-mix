import argparse
import pandas as pd
import os
import numpy as np

path_format = '/k{k}_s{seed}_v{generation_prompt_version}_r{generation_round_idx}/{student_model}/graded/by_{grader_model}_on_text_{column}_{grading_round_idx}_{grading_prompt_version}'

def aggregate_across_generations(score, aggregation_method='average'):
    '''
    aggregate points over multiple generations
    '''
    if aggregation_method=='average':
        score = score.mean(axis=0)
        return score.mean(), score.std(ddof=1) 
    elif aggregation_method=='max':
        score = score.max(axis=0)
        return score.mean(), score.std(ddof=1)
    else:
        raise NotImplementedError
    

def aggregate_points(score, k, max_possible_points, aggregation_method='sum'):
    '''
    aggregate points for one round of generation
    '''
    c = max_possible_points - k
    
    if aggregation_method=='sum':
        return score.sum(axis=2)
    elif aggregation_method=='rescaled_sum':
        return (score.sum(axis=2) / max_possible_points) ** max_possible_points
    elif aggregation_method=='average':
        return score.mean(axis=2)
    # 1 point if all k+3 earned, 0 otherwise
    elif aggregation_method=='all_points_earned_binary':
        score = score.sum(axis=2)
        score = np.where(score == max_possible_points, 1, 0)
        return score    
    # up to 2 possible points 
    # (1 point for getting all first k points, 1 point for getting >= 2 of remaining 3 points)
    elif aggregation_method=='split_skills_vs_rest_binary':
        segments = [slice(0, k), slice(k, max_possible_points)]
        score = [score[:, :, segment].sum(axis=2) for segment in segments]
        score[0] = np.where(score[0] == k, 1, 0)
        score[1] = np.where(score[1] >= (c - 1), 1, 0) 
        return score[0] + score[1]
    # 1 point for getting all first k points and getting >= 2 of remaining 3 points, 0 otherwise
    elif aggregation_method=='all_skills_and_good_rest':
        segments = [slice(0, k), slice(k, max_possible_points)]
        score = [score[:, :, segment].sum(axis=2) for segment in segments]
        score[0] = np.where(score[0] == k, 1, 0)
        score[1] = np.where(score[1] >= (c - 1), 1, 0) 
        return score[0] * score[1]
    # avg over first k points
    elif aggregation_method=='average_over_skills':
        segment = slice(0, k)
        score = score[:, :, segment].mean(axis=2)
        return score
    # sum over first k points
    elif aggregation_method=='sum_over_skills':
        segment = slice(0, k)
        score = score[:, :, segment].sum(axis=2)
        return score
    # avg over the first k points, count as 0 if the other 3 are not perfect
    elif aggregation_method=='average_over_skills_filtered':
        segments = [slice(0, k), slice(k, max_possible_points)]
        score = [score[:, :, segment].sum(axis=2) for segment in segments]
        score = np.where(score[1] == c, score[0]/k, 0)
        return score
    # sum over the first k points, count as 0 if the other 3 are not perfect
    elif aggregation_method=='sum_over_skills_filtered':
        segments = [slice(0, k), slice(k, max_possible_points)]
        score = [score[:, :, segment].sum(axis=2) for segment in segments]
        score = np.where(score[1] == c, score[0], 0)
        return score
    else:
        raise NotImplementedError
        
def aggregate_individual_points(score, aggregation_method='average'):
    '''
    aggregate the grading runs: for each point out of k+3 points
    '''
    # do avg and get a float
    if aggregation_method=='average':
        return score.mean(axis=1)
    elif aggregation_method=='majority_vote':
        return np.rint(score.mean(axis=1))
    else:
        raise NotImplementedError
        
def main(args):
    filter_name = args.filter_explicit_name
    filter_skills = args.filter_skills.split(',')

    student_model_name, grader_model_name = args.student_model_name, args.grader_model_name
    indiv_pt_agg_method = args.indiv_pt_agg_method
    pts_agg_method = args.pts_agg_method.split(',')
    gen_pts_agg_method = args.gen_pts_agg_method
    
    args.k_range = [int(k) for k in args.k_range.split(',')]
    column_headers = [f"k = {k}" for k in args.k_range]
    row = []
    for k in args.k_range:
        max_possible_score = k + 3 
        score = []
        
        for generation_round_idx in range(1, args.num_generation_rounds + 1):
            for grading_round_idx in range(1, args.num_grading_rounds + 1):
                valid_combs = []
                generation_prompt_version = args.generation_prompt_version
                if generation_prompt_version == '-1':
                   generation_prompt_version = 8 if ('llama' in student_model_name) and (k!=2) else 9
                grading_prompt_version = args.grading_prompt_version
                if grading_prompt_version == '-1':
                   grading_prompt_version = 14 if 'llama' in grader_model_name else 10
                
                p = args.exp_path + path_format.format(student_model=student_model_name, grader_model=grader_model_name, column=args.column, k=k, generation_round_idx=generation_round_idx, grading_round_idx=grading_round_idx, generation_prompt_version=generation_prompt_version, grading_prompt_version=grading_prompt_version, seed=args.seed)
                
                try:
                    df = pd.read_csv(os.path.join(p, 'records.csv'))
                except:
                    print(f'No records.csv in {p}')
                    continue
                
                score_cur = []
                for idx, points in enumerate(df['points_0']):
                    points = points.split(',') if type(points)==str else []
                    if len(points) != max_possible_score:
                        if len(points) == 0:
                            points = [0] * (k+3)
                        elif len(points) > max_possible_score:
                            points = points[:max_possible_score]
                        else:
                            # something wrong happened in parsing the scores
                            points = points + [0] * (max_possible_score-len(points))

                    assert len(points) == max_possible_score
                    points[-1] = df['true_sentence_lim_pt_0'][idx]
                    if filter_name:
                        if 'assistant_0' in df.columns:
                            text = df['user_0'][idx]
                        else:
                            text = df['[INST]_0'][idx]
                        text = text.split('For reference, here are the definitions for the skills')[0]
                        text = text.split('The student\'s answer was: ')[1]
                        text = text.strip().lower()
                        skills = df['skills'][idx]
                        skills = skills.split(',')
                        skills = [skill.split('(')[0] for skill in skills]
                        skills = [skill.strip().lower() for skill in skills]
                        for i, skill in enumerate(skills):
                            if skill in text:
                                points[i] = 0
                    
                    skills = df['skills'][idx]
                    skills = [skill.strip() for skill in skills.split(',')]
                    if any(skill in filter_skills for skill in skills):
                        pass
                    else:
                        valid_combs.append(idx)

                    score_cur.append(points)
                
                if len(score_cur) >= args.num_generations_per_round:
                    score_cur = score_cur[:args.num_generations_per_round]
                if len(score_cur) < args.num_generations_per_round:
                    print('not enough generations!')
                    print(len(score_cur), p)
                score.append(score_cur)
            
        if score == []:
            continue
        score = np.array(score).astype(np.float32)
        # reshape to (num_generation_rounds, num_grading_rounds, -1)
        score = score.reshape(args.num_generation_rounds, args.num_grading_rounds, -1, max_possible_score)        
                
        score = aggregate_individual_points(score, aggregation_method=args.indiv_pt_agg_method)
        scores = [aggregate_points(score, k, max_possible_score, aggregation_method=method) for method in pts_agg_method]
        
        print(f'Valid combinations for {k=}:')
        print(valid_combs, len(valid_combs))
        
        results = [aggregate_across_generations(score[:, valid_combs], aggregation_method=args.gen_pts_agg_method) for score in scores]
        n = len(valid_combs)
        score_str = ''
        for result in results:
            mean, std_dev = result
            score_str += "{:.2f} ± {:.3f} ".format(mean, std_dev / np.sqrt(n))
        
        row.append(score_str)

    print(f'{student_model_name=}\n{grader_model_name=}\n\n{indiv_pt_agg_method=}\n{pts_agg_method=}\n{gen_pts_agg_method=}\n')
    for header, row_elem in zip(column_headers, row):
        print(f"[ {header} ][ {row_elem} ]")
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model_name", type=str, default='llama2-70b-chat')
    parser.add_argument("--grader_model_name", type=str, default='gpt-4')

    
    parser.add_argument("--exp_path", type=str, default='test')
    parser.add_argument("--k_range", type=str, default="2,3,4")
    parser.add_argument("--num_generation_rounds", type=int, default=3)
    parser.add_argument("--num_generations_per_round", type=int, default=100)
    
    parser.add_argument("--num_grading_rounds", type=int, default=3)
    
    parser.add_argument("--indiv_pt_agg_method", type=str, default='majority_vote')
    parser.add_argument("--pts_agg_method", type=str, default='all_points_earned_binary,all_skills_and_good_rest,average_over_skills_filtered') #--pts_agg_method=sum,sum_over_skills,rescaled_sum
    parser.add_argument("--gen_pts_agg_method", type=str, default='max')

    parser.add_argument("--filter_explicit_name", action="store_true")
    parser.add_argument("--filter_skills", type=str, default='')
    parser.add_argument("--generation_prompt_version", type=str, default='-1')
    parser.add_argument("--grading_prompt_version", type=str, default='-1')
    parser.add_argument("--seed", type=int, default=10)
    
    parser.add_argument("--column", type=str, default='1')

    args = parser.parse_args()
    main(args) 
