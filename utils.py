import re
import csv
import os

path_to_models_dict = {
        'llama2-7b-chat': "meta-llama/Llama-2-7b-chat-hf",
        'llama2-13b-chat': "meta-llama/Llama-2-13b-chat-hf",
        'llama2-70b-chat': "meta-llama/Llama-2-70b-chat-hf",
        'tigerbot-70b-chat': "TigerResearch/tigerbot-70b-chat", 
        'falcon-180b-chat': "tiiuae/falcon-180B-chat",
        'xwin-lm-70b': 'Xwin-LM/Xwin-LM-70B-V0.1',
        'mistral-7b-chat': 'mistralai/Mistral-7B-Instruct-v0.1', 
        'qwen-14b-chat': "Qwen/Qwen-14B-Chat",
                      }


def get_skills_dict(path='skills.csv'):
    skills_dict = {}
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['status'] not in ['remove', 'in review', 'maybe']:
                row['skill'] = row['skill'].strip()
                skills_dict[row['skill']] = row
    return skills_dict

def load_csv(path, dict_format=True):
    if os.path.exists(path) == False:
        return []
    if dict_format:
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            data = [row for row in reader]
        return data
    else:
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = [row for row in reader]
        return data

def mapping_num_skills_to_num_sentences(num_skills):
    return max(1, num_skills - 1)

def remove_surrounding_quotation_marks(text):
        while text[0]=='"':
            text = text[1:]
        while text[-1]=='"':
            text = text[:-1]
        return text

def remove_non_number_suffix(text):
    while (len(text) > 0) and (text[-1] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
        text = text[:-1]
    return text

def convert_str_to_float(text):
    try:
        return float(text)
    except:
        return 0.0

def manually_extract_points(text, score_marker="earned:"):
    text_list = text.lower().split(score_marker)[1:]
    text_list = [txt.strip().split('\n')[0] for txt in text_list]
    text_list = [txt.split(' ')[0] for txt in text_list if len(txt) > 0]
    text_list = [txt.split('/')[0] for txt in text_list if len(txt) > 0]
    
    points = [convert_str_to_float(remove_non_number_suffix(txt)) for txt in text_list]
    # score = sum(points)
    
    return points

def extract_numerator(text):
    text = text.strip()

    text = re.sub(r"[^0-9./]", "", text) # might not be needed

    match = re.search(r'^(\d+)(?=\/)', text)
    text = match.group(1) if match else text
    if len(text)==0:
        return text
    return float(text)

def modify_dict_key(key):
    if 'total' in key:
        return 'score'
    return key

def create_output_dict(output, delim='|'):
    # print(output)
    #first_idx, last_idx = output.find(delim), output.rfind(delim)
    #output = output[first_idx:last_idx].split('\n')[2:]
    output = output.split('\n')
    output = [row for row in output if delim in row and '--' not in row and 'criteria' not in row.lower()]

    output = [row.split(delim) for row in output]
    output = [row for row in output if row!=[]]
    output = [[elem.strip() for elem in row if len(elem.strip()) > 0] for row in output]
    output = [row for row in output if len(row)>0]

    # print(output)
    output_dict = {row[0].strip().lower(): extract_numerator(row[1]) for row in output if extract_numerator(row[1])!=''}
    output_dict = {modify_dict_key(key): val for key, val in output_dict.items()}

    extracted_score = sum(val for key, val in output_dict.items() if key!='score')
    output_dict['extracted_score'] = extracted_score

    return output_dict

# def count_num_sentences(text):
#     text = re.sub(r'\.{3,}', '...', text)
#     text = re.sub(r'\s*\.{3}', '...', text)
#     text = re.sub(r'\.{3}\s*', '...', text)
    
#     text = re.sub('Mr. ', 'Mr.', text)
#     text = re.sub('Mrs. ', 'Mrs.', text)
#     text = re.sub('Ms. ', 'Ms.', text)
#     text = re.sub('Dr. ', 'Dr.', text)
#     text = re.sub('Prof. ', 'Prof.', text)
    

#     pattern =  "(?<=[.!?])[\s']+|(?<=\.)\Z" #"(?<=[.!?])\s+|(?<=\.)\Z"
   

#     lst = re.split(pattern, text)
#     lst = [txt.strip() for txt in lst]
#     lst = [txt for txt in lst if len(txt)>0]

#     lst =  [elem.split('" ')  if ('" ' in elem and '," ' not in elem) else [elem] for elem in lst]
#     lst = [_elem for elem in lst for _elem in elem]

#     return len(lst)

def count_num_sentences(text):
    text = re.sub('Mr. ', 'Mr.', text)
    text = re.sub('Mrs. ', 'Mrs.', text)
    text = re.sub('Ms. ', 'Ms.', text)
    text = re.sub('Dr. ', 'Dr.', text)
    text = re.sub('Prof. ', 'Prof.', text)

    pattern = r'[.!?][\'\"]* [\'\"]*[A-Z]'
    lst = re.split(pattern, text)
    num_sentences = len(lst)
    return num_sentences


LLAMA_SYS_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""