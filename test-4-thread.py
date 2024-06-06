from typing import Any
from uuid import UUID
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from queue import Queue
from threading import Thread

from langchain_core.outputs import LLMResult


load_dotenv()

queue = Queue()

class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        queue.put(token)
    def on_llm_end(self, response, **kwargs):
        queue.put(None)
    def on_llm_error(self, error, **kwargs):
        queue.put(None)

# TEST CALLBACK HANDLER
chat = ChatOpenAI(
    streaming=True, # it should be True
    callbacks = [StreamingHandler()]
)

prompt = ChatPromptTemplate.from_messages([
    ("human", "{content}")
])

class StreamingChain(LLMChain):
    def stream(self, input):
        def task():
            self(input)

        Thread(target=task).start()

        while True:
            token = queue.get()
            if token is None:
                break
            yield token

chain = StreamingChain(llm=chat, prompt=prompt)

for output in chain.stream(input={"content":"tell me a joke" }):
    print(output)