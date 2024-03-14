try:
    import openai
except:
    print("OpenAI API not installed")
    
import torch

from fastchat.model import load_model, get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
import transformers
import time
import os

class ChatbotEngine():
    def __init__(self, model_path) -> None:
        self.model_path = model_path

    def query(self, conv, **kwargs):
        raise NotImplementedError

    def initialize_conversation(self):
        return get_conversation_template(self.model_path)


class LocalChatbotEngine(ChatbotEngine):
    def __init__(self, model_path, model_args) -> None:
        super().__init__(model_path)
        self.model, self.tokenizer = load_model(
            model_path,
            model_args.device,
            model_args.num_gpus,
            model_args.max_gpu_memory,
            #model_args.load_8bit,
            #model_args.cpu_offloading,
            revision=model_args.revision,
            debug=model_args.debug,
        )
        self.device = model_args.device

    def initialize_conversation(self):
        if 'xwin' in self.model_path.lower():
            conv = get_conversation_template('vicuna-33b')
        else:
            conv = get_conversation_template(self.model_path)
        return conv

    @torch.inference_mode()
    def query(self, conv, temperature=0.7, repetition_penalty=1.0, max_new_tokens=512, **kwargs):
        prompt = conv.get_prompt()
        
        inputs = self.tokenizer([prompt])
        inputs = {k: torch.tensor(v).to(self.device) for k, v in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            do_sample=True if temperature > 1e-5 else False,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs, prompt
    
class MistralChatbotEngine(ChatbotEngine):
    def __init__(self, model_path, model_args) -> None:
        super().__init__(model_path)
        # self.model = AutoModelForCausalLM.from_pretrained(model_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model, self.tokenizer = load_model(
            model_path,
            model_args.device,
            model_args.num_gpus,
            model_args.max_gpu_memory,
            #model_args.load_8bit,
            #model_args.cpu_offloading,
            revision=model_args.revision,
            debug=model_args.debug,
        )
        self.device = model_args.device

    def initialize_conversation(self):
        conv = get_conversation_template('gpt-4')
        return conv

    @torch.inference_mode()
    def query(self, conv, temperature=0.7, repetition_penalty=1.0, max_new_tokens=512, **kwargs):
        messages = [{'role': role, 'content': message} for (role, message) in conv.messages[:-1]]
        # print(messages)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(
            model_inputs, 
            do_sample=True if temperature > 1e-5 else False,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        generated_ids = generated_ids[0][len(encodeds[0]) :]
        generated_ids = generated_ids[:-1]
        outputs = self.tokenizer.decode(generated_ids)
        return outputs, prompt
    
  
class QwenChatbotEngine(ChatbotEngine):
    def __init__(self, model_path, model_args) -> None:
        super().__init__(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参
        self.history = None

    def initialize_conversation(self):
        conv = get_conversation_template('gpt-4')
        self.history = None
        return conv

    @torch.inference_mode()
    def query(self, conv, **kwargs):
        prompt = self.history
        outputs, self.history = self.model.chat(self.tokenizer, conv.messages[-2][1], history=prompt)
        return outputs, prompt
    

class FalconChatbotEngine(ChatbotEngine):

    STOP_SEQUENCES = ["\nUser:", "<|endoftext|>", " User:", "###"]

    def __init__(self, model_path) -> None:
        super().__init__(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        

    def initialize_conversation(self):
        conv = get_conversation_template(self.model_path)
        conv.roles = (conv.roles[0], "Falcon")
        return conv

    def query(self, conv, temperature=0.7, repetition_penalty=1.0, max_new_tokens=512, **kwargs):
        prompt = conv.get_prompt()
        
        sequences = self.pipeline(
            prompt, 
            return_full_text=False,
            do_sample=True if temperature > 1e-5 else False,
            temperature=temperature, 
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens, 
            num_return_sequences=1, 
            eos_token_id=self.tokenizer.eos_token_id,
        )
        outputs = sequences[0]['generated_text']
        for stop_sequence in self.STOP_SEQUENCES:
            if stop_sequence in outputs:
                outputs = outputs.split(stop_sequence)[0]
        return outputs, prompt


class OpenAIChatbotEngine(ChatbotEngine):
    def __init__(self, model_path) -> None:
        super().__init__(model_path)
        if "OPENAI_API_KEY" not in os.environ:
            raise Exception("OPENAI_API_KEY not found, please set it in your environment variables")
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.sleep_time = 10
        self.failed_attempts = 0
    
    def query(self, conv, temperature=0.7, repetition_penalty=1, max_new_tokens=512, nruns=1, **kwargs):
        prompt = conv.to_openai_api_messages()
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model = self.model_path,
                    messages = prompt,
                    n=nruns,
                    max_tokens=max_new_tokens,
                )
                self.sleep_time = 10
                self.failed_attempts = 0
                break
            except Exception as e:
                print(e)
                # sleep for 10 seconds
                time.sleep(self.sleep_time)
                self.sleep_time *= 2
                self.failed_attempts += 1
                if self.failed_attempts > 10:
                    raise Exception("Too many failed attempts")
        outputs = [choice.message.content for choice in response.choices]
        if len(outputs) == 1:
            outputs = outputs[0]
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        print(f"Cost so far: {(0.03 * self.total_prompt_tokens + 0.06 * self.total_completion_tokens) / 1000} USD")
        return outputs, prompt
    
class HumanEngine(ChatbotEngine):
    def __init__(self, model_path):
        super().__init__(model_path)
    
    def query(self, conv, **kwargs):
        question = conv.messages[-2][1]
        outputs = ""
        return outputs, question

def get_engine(model_path, args):
    if 'gpt' in model_path:
        engine = OpenAIChatbotEngine(model_path)
    elif model_path in ['human']:
        engine = HumanEngine(model_path)
    elif 'falcon' in model_path:
        engine = FalconChatbotEngine(model_path)
    elif 'mistral' in model_path.lower():
        engine = MistralChatbotEngine(model_path, args)
    elif 'qwen' in model_path.lower():
        engine = QwenChatbotEngine(model_path, args)
    else:
        engine = LocalChatbotEngine(model_path, args)
    return engine
