from functools import partial
from .chatopenai import build_llm

llm_map = {
    "gpt-4": partial(build_llm, model_name='gpt-4'),
    "gpt-3.5-turbo": partial(build_llm, model_name='gpt-3.5-turbo')
}	

# Behind the scenes:
# builder = llm_map['gpt-4']
# builder(chat_args) # llm that uses gpt-4
# 
# or
# builder = llm_map['gpt-3.5-turbo']
# builder(chat_args, model_name='gpt-3.5-turbo') 