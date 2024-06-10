from langchain.chat_models import ChatOpenAI

def build_llm(chat_args, model_name):
    return ChatOpenAI(
        streaming=chat_args.streaming,
        # model_name='gpt-3.5-turbo', # default model
        # model_name='gpt-4',
        model_name=model_name
    )