import transformers as trans
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer
from openue.data.utils import get_labels_ner, get_labels_seq, OutputExample
from typing import Dict

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
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # import pdb; pdb.set_trace()
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

        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        relation_output = self.relation_classification(cls_output)
        relation_output_sigmoid = torch.sigmoid(relation_output)

        if label_ids_seq is None:
            return (relation_output_sigmoid, relation_output, cls_output)
        else:
            # 跟label算个loss
            Loss = torch.nn.BCELoss()

            loss = Loss(relation_output_sigmoid, label_ids_seq)

            return (loss, relation_output_sigmoid, relation_output, cls_output)

    def add_to_argparse(parser):
        parser.add_argument("--model_type", type=str, default="bert")



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
        output_hidden_states=None,
        return_dict=None,
        label_ids_seq=None,
        label_ids_ner=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs = {}
        inputs['input_ids'] = input_ids
        inputs['attention_mask'] = attention_mask
        inputs['token_type_ids'] = token_type_ids

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

        # batch_size * 107 * hidden_size
        sequence_poolout_output = outputs[0]
        # batch_size * 107 * 6
        logits = self.token_classification(sequence_poolout_output)

        if label_ids_ner is None:
            return logits ,outputs[1]

        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, label_ids_ner.view(-1), torch.tensor(loss_fct.ignore_index).type_as(label_ids_ner)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), label_ids_ner.view(-1))

        # if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    
    
    def add_to_argparse(parser):
        parser.add_argument("--model_type", type=str, default="bert")


class Inference(torch.nn.Module):
    """
        input the text, 
        return the triples
    """
    def __init__(self, args):
        self.args = args
        # init the labels
        self._init_labels()    
        self._init_models()
        
        
        self.mode = "event" if "event" in args.task_name else "triple"
        
        if self.mode == "event":
            self.process = self.event_process
        else:
            self.process = self.normal_process
        
     
    
    def _init_labels(self):
        self.labels_ner = get_labels_ner()
        self.label_map_ner: Dict[int, str] = {i: label for i, label in enumerate(self.labels_ner)}
        num_labels_ner = len(self.labels_ner)

        # 读取seq的label
        self.labels_seq = get_labels_seq()
        self.label_map_seq: Dict[int, str] = {i: label for i, label in enumerate(self.labels_seq)}
        num_labels_seq = len(self.labels_seq)
    
    
    def _init_models(self):
        model_name_or_path = self.args.seq_model_name_or_path
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            # './vocab.txt',
            num_labels=self.num_labels_seq,
            # id2label=label_map_seq,
            label2id={label: i for i, label in enumerate(self.labels_ner)},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
        )
        self.model_seq = BertForRelationClassification.from_pretrained(
            model_name_or_path,
            # from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )

        model_name_or_path = self.args.ner_model_name_or_path
        # 读取待训练的ner模型
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels_ner,
            id2label=self.label_map_ner,
            label2id={label: i for i, label in enumerate(self.labels_ner)},
        )
        # tokenizer = BertTokenizer.from_pretrained(
        #     model_name_or_path,
        #     cache_dir=model_args.cache_dir,
        #     use_fast=model_args.use_fast,
        # )
        self.model_ner = BertForNER.from_pretrained(
            model_name_or_path,
            config=config,
        )

    def forward(self, inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        inputs_seq = {'input_ids': inputs['input_ids_seq'],
                    'token_type_ids': inputs['token_type_ids_seq'],
                    'attention_mask': inputs['attention_mask_seq'],
                    # 'label_ids_seq': inputs['label_ids_seq']
                    }

        with torch.no_grad():
            outputs_seq = self.model_seq(**inputs_seq)

            batch_size = inputs_seq['input_ids'].shape[0]
            num_relations = len(self.label_map_seq.keys())
            max_length = inputs_seq['input_ids'].shape[1]

            # [batch_size, 50]
            # relation_output_sigmoid = outputs_seq[1]
            relation_output_sigmoid = outputs_seq[0]

            # 多关系预测
            relation_output_sigmoid_ = relation_output_sigmoid > 0.5
            # # 这个0.5是超参数，超参数
            if torch.sum(relation_output_sigmoid_).tolist() == 0:
                idx = torch.max(relation_output_sigmoid, dim=1)[1].tolist()[0]
                relation_output_sigmoid_[0][idx] = 1

            # [batch_size, 50]
            relation_output_sigmoid_ = relation_output_sigmoid_.long()
            # [batch_size * 50, ]
            relation_output_sigmoid_index = relation_output_sigmoid_.view(-1)

            index_ = torch.arange(0, num_relations).to(self.device)
            index_ = index_.expand(batch_size, num_relations)
            # 需要拼接的部分1：REL
            relation_output_sigmoid_number = torch.masked_select(index_, relation_output_sigmoid_.bool())
            # 需要拼接的部分2：SEP
            cat_sep = torch.full((relation_output_sigmoid_number.shape[0], 1), 102).long().to(self.device)
            # 需要拼接的部分3：[1]
            cat_one = torch.full((relation_output_sigmoid_number.shape[0], 1), 1).long().to(self.device)
            # 需要拼接的部4：[0]
            cat_zero = torch.full((relation_output_sigmoid_number.shape[0], 1), 0).long().to(self.device)

            # 拼接input_ids_seq的输入
            input_ids_ner = torch.unsqueeze(inputs['input_ids_seq'], 1)
            # [batch_size, 50, max_length], 复制50份
            input_ids_ner = input_ids_ner.expand(-1, len(self.label_map_seq.keys()), -1)
            # [batch_size * 50, max_length]
            input_ids_ner_reshape = input_ids_ner.reshape(batch_size * num_relations, max_length)
            # 选择预测正确的所有关系
            tmp1 = relation_output_sigmoid_index.unsqueeze(dim=1)  # [200, 1]
            mask = tmp1.expand(-1, max_length)  # [200, 79]
            tmp2 = torch.masked_select(input_ids_ner_reshape, mask.bool())
            # n(选出来的关系数字) * max_length
            # n >> batch_size, 因为一句话中有多个关系
            tmp3 = tmp2.view(-1, max_length)
            # 拼接 0
            tmp4 = torch.cat((tmp3, cat_zero), 1)
            # 拼接 0
            input_ids_ner = torch.cat((tmp4, cat_zero), 1)

            # 利用attention中1的求和的到rel_pos的位置
            attention_mask_ner = torch.unsqueeze(inputs['attention_mask_seq'], 1)
            # [batch_size, 50, max_length], 复制50份
            attention_mask_ner = attention_mask_ner.expand(-1, len(self.label_map_seq.keys()), -1)
            # [batch_size * 50, max_length]
            attention_mask_ner_reshape = attention_mask_ner.reshape(batch_size * num_relations, max_length)
            # 选择预测正确的所有关系
            tmp1 = relation_output_sigmoid_index.unsqueeze(dim=1)  # [200, 1]
            mask = tmp1.expand(-1, max_length)  # [200, 79]
            tmp2 = torch.masked_select(attention_mask_ner_reshape, mask.bool())
            # n(选出来的关系数字) * max_length
            # n >> batch_size, 因为一句话中有多个关系
            tmp3 = tmp2.view(-1, max_length)
            # 利用attention中1的求和的到rel_pos的位置
            rel_pos = torch.sum(tmp3, dim=1)
            (rel_number_find, max_length_find) = input_ids_ner.shape
            one_hot = torch.sparse.torch.eye(max_length_find).long().to(self.device)
            rel_pos_mask = one_hot.index_select(0, rel_pos)
            rel_pos_mask_plus = one_hot.index_select(0, rel_pos+1)

            # 拼接input_ids的输入
            input_ids_ner[rel_pos_mask.bool()] = relation_output_sigmoid_number
            input_ids_ner[rel_pos_mask_plus.bool()] = cat_sep.squeeze()

            # 拼接token_type_ids的输入
            token_type_ids_ner = torch.zeros(rel_number_find, max_length_find).to(self.device)
            token_type_ids_ner[rel_pos_mask.bool()] = 1
            token_type_ids_ner[rel_pos_mask_plus.bool()] = 1
            token_type_ids_ner = token_type_ids_ner.long()

            # 拼接attention_mask的输入
            # 拼接 0
            tmp4 = torch.cat((tmp3, cat_zero), dim=1)
            # 拼接 0
            tmp5 = torch.cat((tmp4, cat_zero), dim=1)
            tmp5[rel_pos_mask.bool()] = 1
            tmp5[rel_pos_mask_plus.bool()] = 1
            attention_mask_ner_tmp = tmp5

            inputs_ner = {'input_ids': input_ids_ner,
                        'token_type_ids': token_type_ids_ner,
                        'attention_mask': attention_mask_ner_tmp,
                        # 'label_ids_ner': inputs['label_ids_ner'].long()
                        }

            try:
                outputs_ner = self.model_ner(**inputs_ner)[0]
            except BaseException:
                print('23')

            _, results = torch.max(outputs_ner, dim=2)
            results_np = results.cpu().numpy()
            attention_position_np = rel_pos.cpu().numpy()

            results_list = results_np.tolist()
            attention_position_list = attention_position_np.tolist()
            predict_relation_list = relation_output_sigmoid_number.long().tolist()
            input_ids_list = input_ids_ner.tolist()

            processed_results_list = []
            processed_input_ids_list = []
            for idx, result in enumerate(results_list):
                tmp1 = result[0: attention_position_list[idx]-1]
                tmp2 = input_ids_list[idx][0: attention_position_list[idx]-1]
                processed_results_list.append(tmp1)
                processed_input_ids_list.append(tmp2)

            processed_results_list_BIO = []
            for result in processed_results_list:
                processed_results_list_BIO.append([self.label_map_ner[token] for token in result])

            # 把结果剥离出来
            index = 0
            triple_output = []

            # for each relation type or event type
            if self.mode == "triple":
                for ids, BIOS in zip(processed_input_ids_list, processed_results_list_BIO):
                    labels = self.process(ids, BIOS)
                    # r = label_map_seq[predict_relation_list[index]]
                    r = predict_relation_list[index]

                    if len(labels['subject']) == 0:
                        h = None
                    else:
                        h = labels['subject'][0]
                        # h = ''.join(tokenizer.convert_ids_to_tokens(h))

                    if len(labels['object']) == 0:
                        t = None
                    else:
                        t = labels['object'][0]
                        # t = ''.join(tokenizer.convert_ids_to_tokens(t))

                    triple_output.append(OutputExample(h=h, r=r, t=t))

                    index = index + 1
            elif self.mode == "event":
                for ids, BIOS in zip(processed_input_ids_list, processed_results_list_BIO):
                    triple_output.append(dict(event_type=predict_relation_list[index], argument=self.process(ids, BIOS)))

            return triple_output, inputs['label_ids']
        
    @staticmethod 
    def normal_process(text, result):
        index = 0
        start = None
        labels = {}
        labels['subject'] = []
        labels['object'] = []
        indicator = ''
        for w, t in zip(text, result):
            # ["O", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "Relation"
            if start is None:
                if t == 'B-SUB':
                    start = index
                    indicator = 'subject'
                elif t == 'B-OBJ':
                    start = index
                    indicator = 'object'
            else:
                # if t == 'I-SUB' or t == 'I-OBJ':
                #     continue
                if t == "O":
                    # print(result[start: index])
                    labels[indicator].append(text[start: index])
                    start = None
            index += 1
        # print(labels)
        return labels
    
    
    @staticmethod 
    def event_process(text, result):
        """
        return List[Dict(text, label)]
        """
        index = 0
        start = None
        labels = []
        indicator = ''
        for w, t in zip(text, result):
            # ["O", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "Relation"
            if start is None:
                if "B-" in t:
                    # get the label name
                    indicator = t.split("-")[-1]
                    start = index
            else:
                if t.split("-")[-1] != indicator or "B-" in t:
                    # B-a I-b wrong, B-a B-a wrong
                    start = None
                elif t == "O":
                    # print(result[start: index])
                    labels.append(dict(text=text[start: index], label=indicator))
                    start = None
            index += 1
        # print(labels)
        return labels
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--seq_model_name_or_path", type=str, default="seq_model")
        parser.add_argument("--ner_model_name_or_path", type=str, default="ner_model")
        
        return parser