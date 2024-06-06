from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv


load_dotenv()

class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        # print(token)
        pass

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
        print(self(input))
        yield 'hi'
        yield 'there'

chain = StreamingChain(llm=chat, prompt=prompt)

# chain.stream('whatever')
# print(chain('tell me a joke'))

# for output in chain.stream('whatever'):
#     print(output)

for output in chain.stream(input={"content":"tell me a joke" }):
    print(output)