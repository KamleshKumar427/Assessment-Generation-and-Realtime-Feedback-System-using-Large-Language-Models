
import json
import os
from pprint import pprint
import warnings

import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login, HfFolder
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import necessary libraries
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain

print("done importing")


# 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


from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

from huggingface_hub import HfApi
api = HfApi()


warnings.filterwarnings("ignore")

# Loading the model and tokenizer
def Load_model():
    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    generation_config = model.generation_config
    generation_config.max_new_tokens = 100
    generation_config.temperature = 0.7
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id



    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=10000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    return pipeline



pipeline = Load_model()

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0.7})

# Extra Code
# def langchain_setup():
#     teacher_prompt_template = """<s>[INST] <<SYS>>
#     {{ You are a teacher, Teaching student about some topic and then student will ask you some questions and you have to answer these as a teacher.}}<<SYS>>
#     ###

#     Previous Conversation:
#     '''
#     {history}
#     '''

#     {{{input}}}[/INST]

#     """
#     prompt = PromptTemplate(template=teacher_prompt_template, input_variables=['input', 'history'])

#     chain = ConversationChain(llm=llm, prompt=prompt)

#     return chain

# chain = langchain_setup()

# def Model_interaction(param1: str):
#     teacher_response = chain.run(f"{param1}.")
#     return teacher_response


# teacher_response = Model_interaction("Teach me parts of speech in details, teacher")
# print(teacher_response)


# assesment code
Assesment_prompt_template = """<s>[INST] <<SYS>>
{{ You are a teacher, producing assignments for students and providing the based on the assignment results}}<<SYS>>
###

Previous Conversation:
'''
{history}
'''

{{{input}}}[/INST]

"""
prompt = PromptTemplate(template=Assesment_prompt_template, input_variables=['input', 'history'])

chain = ConversationChain(llm=llm, prompt=prompt)

def generate_assesment(Topic):
    Assesment = chain.run(f"Generate 3 multiple choice questions and 2 short written questions  only about {Topic}. Don't give answers only questions")
    return Assesment

# You are a teacher, producing feedback the based on the assignment results. If MCQS answer is wrong mention it in the feedback.

# Feedback Code
feedback_prompt_template = """<s>[INST] <<SYS>>
{{Your task is to generate feedback for students based on their assignment. This assignment includes multiple-choice questions (MCQs) among showrt answers of questions. When reviewing the assignment, pay special attention to the MCQs. For each MCQ, if a student has selected an incorrect/don't know answer, your feedback should mention the wrong answers/and answers not given. }}<<SYS>>
###


Previous Conversation:
'''
{history}
'''

{{{input}}}[/INST]

"""

feedback_prompt = PromptTemplate(template=feedback_prompt_template, input_variables=['input', 'history'])

feedback_chain = ConversationChain(llm=llm, prompt=feedback_prompt)

def generate_feedback(User_resposponse):
    actual_feeback = feedback_chain.run(User_resposponse)    
    return actual_feeback


readMaterial_prompt_template = """<s>[INST] <<SYS>>
{{ Provide Reading material based on your feedback, which student can use to study to improve their grades.}}<<SYS>>
###

Previous Conversation:
'''
{history}
'''

{{{input}}}[/INST]

"""


readMaterial_prompt = PromptTemplate(template=readMaterial_prompt_template, input_variables=['input', 'history'])

readMaterial_chain = ConversationChain(llm=llm, prompt=readMaterial_prompt)

def generate_readMaterial(actual_feeback):
    actual_readMaterial = readMaterial_chain.run(actual_feeback)    
    return actual_readMaterial


# Server Code

class AssessmentRequest(BaseModel):
    # For user's selection of topic
    param1: str

# extra
class feedbackrequest(BaseModel):
    # For user's selection of topic
    param1: str

class StudyMaterialequest(BaseModel):
    # For user's selection of topic
    param1: str

@app.post("/interact_with_teacher/")
async def assesment_response_endpoint(request: AssessmentRequest):
    result = generate_assesment(request.param1)
    return {"result": result}


@app.post("/feedback/")
async def feedback_response_endpoint(request: feedbackrequest):
    result = generate_feedback(request.param1)
    return {"result": result}

@app.post("/StudyMaterial/")
async def material_response_endpoint(request: StudyMaterialequest):
    result = generate_readMaterial(request.param1)
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
