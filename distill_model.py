import transformers as trans
import torch
from torch.nn import CrossEntropyLoss, MSELoss, KLDivLoss
from types import SimpleNamespace

class BertForRelationClassification(trans.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = trans.BertModel(config)
        self.relation_classification = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        label_ids_seq=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden = outputs.hidden_states
        
        # batch_size * ? * hidden_size
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        relation_output = self.relation_classification(cls_output)
        relation_output_sigmoid = torch.sigmoid(relation_output)

        output_dict = {"logits": relation_output_sigmoid, "hidden": hidden}

        return SimpleNamespace(**output_dict)

class TinyBertForRelationClassification(trans.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = trans.BertModel(config)
        self.relation_classification = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        label_ids_seq=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden = outputs.hidden_states

        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        relation_output = self.relation_classification(cls_output)
        relation_output_sigmoid = torch.sigmoid(relation_output)
        # import pdb; pdb.set_trace()
        output_dict = {"logits": relation_output_sigmoid, "hidden": hidden}

        return SimpleNamespace(**output_dict)


class BertForNER(trans.BertPreTrainedModel):

    def __init__(self, config, **model_kwargs):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.bert = trans.BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.token_classification = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        # labels=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=True,
        label_ids_seq=None,
        label_ids_ner=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden = outputs.hidden_states

        sequence_poolout_output = outputs[0]
        logits = self.token_classification(sequence_poolout_output)

        output_dict = {"logits": logits, "hidden": hidden}

        return SimpleNamespace(**output_dict)

class TinyBertForNER(trans.BertPreTrainedModel):

    def __init__(self, config, **model_kwargs):
        super().__init__(config)

        self.num_labels = config.num_labels

        # self.bert_SEQ = model_kwargs['trained_model']
        # self.label_map_seq = model_kwargs['label_map_seq']
        # self.label_map_ner = model_kwargs['label_map_ner']

        self.bert = trans.BertModel(config)

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.token_classification = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        # labels=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=True,
        label_ids_seq=None,
        label_ids_ner=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden = outputs.hidden_states

        sequence_poolout_output = outputs[0]
        logits = self.token_classification(sequence_poolout_output)
        
        output_dict = {"logits": logits, "hidden": hidden}

        return SimpleNamespace(**output_dict)