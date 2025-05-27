import json
import numpy as np
import torch
from transformers import pipeline
from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer
from awq import AutoAWQForCausalLM

MODEL_NAME = "TheBloke/zephyr-7B-beta-AWQ"

class InferlessPythonModel:

    def initialize(self):
        self.model = AutoAWQForCausalLM.from_quantized(MODEL_NAME, fuse_layers=False, version="GEMV")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def infer(self, inputs, stream_output_handler):

        prompt = inputs["TEXT"]
        messages = [{ "role": "system", "content": "You are an agent that know about about cooking." }] 
        messages.append({ "role": "user", "content": prompt })
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
        generation_kwargs = dict(
            inputs=tokenized_chat,
            streamer=self.streamer,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.2,
            max_new_tokens=1024,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in self.streamer:
            output_dict = {}
            output_dict["OUT"] = new_text
            stream_output_handler.send_streamed_output(output_dict)
        thread.join()

        stream_output_handler.finalise_streamed_output()



    # perform any cleanup activity here
    def finalize(self):
        self.pipe = None
