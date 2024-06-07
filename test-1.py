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
chain = LLMChain(llm=chat, prompt=prompt)

for output in chain.stream(input={"content":"tell me a joke" }):
    print(output)



# TEST CHAIN AND CHAT
# chat = ChatOpenAI(streaming=True)
# chat = ChatOpenAI(streaming=False) # If I change streaming=True to False, it doesn't matter because I'm trying to call my language model with the stream method and that is going to override streaming of False.

# prompt = ChatPromptTemplate.from_messages([
#     ('human', '{content}')
# ])

# TEST CHAIN OUTPUT
# chain = LLMChain(llm = chat, prompt = prompt)

# subtest 1
# output = chain('tell me a joke')
# print(output)

# subtest 2
# output = chain.stream(
#     input = {'content':'tell me a joke'}
# )
# print(output) # it returns a generator object. A generator is kind of like a for loop that allows us to receive little chunks of information over time.

# subtest 3
# for  output in chain.stream(
#     input = {'content':'tell me a joke'}
# ):
#     print(output)


# TEST CHAT OUTPUT
messages = prompt.format_messages(
    content = 'tell me a joke'
)

# subtest 1
# output = chat(messages) # == chat.__call__(messages)
# print(output)

# subtest 2
# output = chat.invoke(messages) # it is similar to chat(messages) and chat.__call__(messages)
# print(output)

# subtest 3
# output = chat.stream(messages) # it returns a generator object. A generator is kind of like a for loop that allows us to receive little chunks of information over time.
# print(output)

# for message in chat.stream(messages):
#     print(message)