from typing import Any, Dict, List
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage

class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue
        self.streaming_run_id = set()

    def on_chat_model_start(self, serialized, messages, run_id, **kwargs):
        if serialized['kwargs']['streaming']:
            # print('This is a streaming model! I should listen to events with a run_id of ', run_id)
            self.streaming_run_id.add(run_id)

    def on_llm_new_token(self, token, **kwargs):
        self.queue.put(token)

    def on_llm_end(self, response, run_id, **kwargs):
        if run_id in self.streaming_run_id:
            self.queue.put(None)
            self.streaming_run_id.remove(run_id)

    def on_llm_error(self, error, **kwargs):
        self.queue.put(None)