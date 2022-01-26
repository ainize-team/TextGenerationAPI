import gc
import json

import torch
from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger


class TextGenerationHandler(BaseHandler):
    def __init__(self):
        logger.info("Torch Serve Start")
        super(TextGenerationHandler, self).__init__()
        self.model_max_length = 512
        self.model = None
        self.tokenizer = None
        self.device = None
        self.task_config = None
        self.initialized = False

    def load_model(self, model_dir):
        logger.info("Load Model Start")
        if self.device.type == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", low_cpu_mem_usage=True)
            self.model.to(self.device, non_blocking=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto")
        if hasattr(self.model.config, "n_positions"):
            self.model_max_length = self.model.config.n_positions
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.model_max_length = self.model.config.max_position_embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logger.info("Load Model Done")

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )
        self.load_model(model_dir)
        self.model.eval()
        self.initialized = True

    def preprocess(self, requests):
        return {
            "text_inputs": requests[0].get("body").get("text_inputs"),
            "max_length": requests[0].get("body").get("max_length", None),
            "min_length": requests[0].get("body").get("min_length", None),
            "do_sample": requests[0].get("body").get("do_sample", None),
            "early_stopping": requests[0].get("body").get("early_stopping", None),
            "num_beams": requests[0].get("body").get("num_beams", None),
            "temperature": requests[0].get("body").get("temperature", None),
            "top_k": requests[0].get("body").get("top_k", None),
            "top_p": requests[0].get("body").get("top_p", None),
            "repetition_penalty": requests[0].get("body").get("repetition_penalty", None),
            "length_penalty": requests[0].get("body").get("length_penalty", None),
            "no_repeat_ngram_size": requests[0].get("body").get("no_repeat_ngram_size", None),
            "num_return_sequences": requests[0].get("body").get("num_return_sequences", None)
        }

    def inference(self, data):
        if len(data["text_inputs"]) >= self.model_max_length * 128:
            logger.warning("The user's input was modified because the user's input was too long.")
            data["text_inputs"] = data["text_inputs"][:self.model_max_length * 128]
        inputs = self.tokenizer.encode(data["text_inputs"], return_tensors="pt")
        if inputs.shape[1] > self.model_max_length:
            logger.warning(f"encoded sequence length({inputs.shape[1]}) is too long.")
            inputs = inputs[:, :self.model_max_length]
        if self.device.type == "cuda":
            inputs = inputs.to(self.device, non_blocking=True)
        data["inputs"] = inputs
        if data["max_length"] and data["max_length"] > self.model_max_length:
            logger.warning(f"change max_length from {data['max_length']} to {self.model_max_length}")
            data["max_length"] = self.model_max_length
        if data["min_length"] and data["min_length"] > self.model_max_length:
            logger.warning(f"change max_length from {data['min_length']} to {self.model_max_length}")
            data["min_length"] = self.model_max_length
        if data["min_length"] and data["max_length"] and data["min_length"] > data["max_length"]:
            logger.warning(f"change min_length from {data['min_length']} to {data['max_length']}")
            data["min_length"] = data['max_length']
        inference_output = self.model.generate(**data).to("cpu").tolist()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return inference_output

    def postprocess(self, inference_output):
        output = self.tokenizer.batch_decode(inference_output, skip_special_tokens=True)
        del inference_output
        gc.collect()
        return [json.dumps(output, ensure_ascii=False)]

    def handle(self, data, context):
        self.context = context
        data = self.preprocess(data)
        data = self.inference(data)
        data = self.postprocess(data)
        return data
