from abc import ABC
import enum
import json
import logging
import os
import ast
from posixpath import realpath
import torch
import transformers
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
)
from transformers.models.bert.configuration_bert import BertConfig
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s",transformers.__version__)


from model import BertForNER, BertForRelationClassification


class BertForSEQHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
	
    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithmfor Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")

        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(model_pt_path, map_location=self.device)
        elif self.setup_config["save_mode"] == "pretrained":
            # 载入推理模型
            self.model = BertForRelationClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            
        else:
            logger.warning("Missing the checkpoint or state_dict.")

        # 载入tokenizer
        if any(fname for fname in os.listdir(model_dir) if fname.startswith("vocab.") and os.path.isfile(fname)):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, do_lower_case=self.setup_config["do_lower_case"]
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.setup_config["model_name"],
                do_lower_case=self.setup_config["do_lower_case"],
            )

        self.model.eval()

        logger.info(
            "Transformer model from path %s loaded successfully", model_dir
        )

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        # Question answering does not need the index_to_name.json file.
        if not self.setup_config["mode"] == "question_answering":
            if os.path.isfile(mapping_file_path):
                with open(mapping_file_path) as f:
                    self.mapping = json.load(f)
            else:
                logger.warning("Missing the index_to_name.json file.")
        self.initialized = True
    
    def preprocess(self, requests):
        # convert the input_ids to (input_ids, attention_mask, token_type_ids)
        # receive {inputs: {input_ids: xxx}}
        total_inputs = []
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode('utf-8')
            max_length = int(self.setup_config["max_length"])
            json_input = input_text
            logger.info(f"Received text: {input_text}")
            inputs = json_input
            
            total_inputs.append(inputs)
        # pack the batch
        inputs = dict(
            input_ids=torch.tensor([_['input_ids'] for _ in total_inputs]).to(self.device),
            attention_mask=torch.tensor([_['attention_mask'] for _ in total_inputs]).to(self.device),
            token_type_ids=torch.tensor([_['token_type_ids'] for _ in total_inputs]).to(self.device),
        )

        return inputs

    def inference(self, inputs):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        # logger.info(inputs)
        # sigmoid output 
        for k, v in inputs.items():
            if len(v.shape) == 3:
                inputs[k] = v.squeeze(1)
        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])
        # Handling inference for sequence_classification.
        # logger.info(outputs)
        return outputs[0]

    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.

        Args:
            data (Torch Tensor): The torch tensor received from the prediction output of the model.

        Returns:
            List: The post process function returns a list of the predicted output.
        """

        return [dict(outputs=d.tolist()) for d in data]

