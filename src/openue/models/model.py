import transformers as trans
import torch
import pytorch_lightning as pl
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
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
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
            loss = self.loss_fn(relation_output, label_ids_seq)

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
        sequence_poolout_output = self.dropout(outputs[0])
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


class Inference(pl.LightningModule):
    """
        input the text, 
        return the triples
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        # init the labels
        self._init_labels()    
        self._init_models()
        
        
        self.mode = "event" if "event" in args.task_name else "triple"
        self.start_idx = self.tokenizer("[relation0]", add_special_tokens=False)['input_ids'][0]
        
        if self.mode == "event":
            self.process = self.event_process
        else:
            self.process = self.normal_process
        
    
    def _init_labels(self):
        self.labels_ner = get_labels_ner()
        self.label_map_ner: Dict[int, str] = {i: label for i, label in enumerate(self.labels_ner)}
        self.num_labels_ner = len(self.labels_ner)

        # 读取seq的label
        self.labels_seq = get_labels_seq(self.args)
        self.label_map_seq: Dict[int, str] = {i: label for i, label in enumerate(self.labels_seq)}
        self.num_labels_seq = len(self.labels_seq)

    
    
    def _init_models(self):
        model_name_or_path = self.args.seq_model_name_or_path
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels_seq,
            label2id={label: i for i, label in enumerate(self.labels_seq)},
        )
        
        self.model_seq = BertForRelationClassification.from_pretrained(
            model_name_or_path,
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
        self.model_ner = BertForNER.from_pretrained(
            model_name_or_path,
            config=config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
        )

    def forward(self, inputs):
        """
        两种方案，一种直接所有relation搞起来，一种使用动态batch size, 针对出现的relation进行forward
        首先通过model_seq获得输入语句的类别标签，batch中每一个样本中含有的关系，
        之后选择大于阈值(0.5)的关系，将其输入取出来得到[batch_size*num_relation, seq_length]的输入向量，以及每一个样本对应的关系数量，
        将其增加了关系类别embedding之后，输入到model_ner中，得到input_ids中每一个token的类别，之后常规的实体识别。
        
        """
        # for k, v in inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         inputs[k] = v.to(self.device)

        inputs_seq = {'input_ids': inputs['input_ids'],
                    'token_type_ids': inputs['token_type_ids'],
                    'attention_mask': inputs['attention_mask'],
                    }

        with torch.no_grad():
            outputs_seq = self.model_seq(**inputs_seq)

            batch_size = inputs_seq['input_ids'].shape[0]
            num_relations = len(self.label_map_seq.keys())
            max_length = inputs_seq['input_ids'].shape[1]

            # [batch_size, num_relation]
            relation_output_sigmoid = outputs_seq[0]

            # 多关系预测
            mask_relation_output_sigmoid = relation_output_sigmoid > 0.5
            # # 这个0.5是超参数，超参数
            # 如果没有关系那就选一个最大概率的关系抽取。
            for i in range(batch_size):
                if torch.sum(mask_relation_output_sigmoid[i]) == 0:
                    max_relation_idx = torch.max(relation_output_sigmoid[i], dim=0)[1].item()
                    mask_relation_output_sigmoid[i][max_relation_idx] = 1

            mask_relation_output_sigmoid = mask_relation_output_sigmoid.long()
            # mask_output [batch_size*num_relation] 表示哪一个输入是需要的
            mask_output = mask_relation_output_sigmoid.view(-1)

            # relation 特殊表示，需要拼接 input_ids :[SEP relation]  attention_mask: [1 1] token_type_ids:[1 1]
            # relation_index shape : [batch_size, num_relations]
            relation_index = torch.arange(self.start_idx, self.start_idx+num_relations).to(self.device).expand(batch_size, num_relations)
            # 需要拼接的部分1：REL， 选取拼接的部分 [batch_size * xxx 不定]
            relation_ids = torch.masked_select(relation_index, mask_relation_output_sigmoid.bool())
            # 需要拼接的部分2：SEP
            cat_sep = torch.full((relation_ids.shape[0], 1), 102).long().to(self.device)
            # 需要拼接的部分3：[1]
            cat_one = torch.full((relation_ids.shape[0], 1), 1).long().to(self.device)
            # 需要拼接的部4：[0]
            cat_zero = torch.full((relation_ids.shape[0], 1), 0).long().to(self.device)

            # 需要原来的input_ids 扩展到relation num维度。
            input_ids_ner = torch.unsqueeze(inputs['input_ids'], 1) # [batch_size, 1, seq_length]
            # [batch_size, 50, max_length], 复制50份
            input_ids_ner = input_ids_ner.expand(-1, len(self.label_map_seq.keys()), -1)
            # [batch_size * 50, max_length]
            input_ids_ner_reshape = input_ids_ner.reshape(batch_size * num_relations, max_length)
            # 选择预测正确的所有关系
            mask = mask_output.unsqueeze(dim=1).expand(-1, max_length)  # [batch_size * num_relations, max_length]
            # 选取了正确的input_ids
            input_ids = torch.masked_select(input_ids_ner_reshape, mask.bool()).view(-1, max_length)
            # n(选出来的关系数字) * max_length
            # n >> batch_size, 因为一句话中有多个关系
            # 添加 sep relation_ids 需要增加的东西
            input_ids = torch.cat((input_ids, cat_zero), 1)
            input_ids_ner = torch.cat((input_ids, cat_zero), 1)

            # 利用attention中1的求和的到rel_pos的位置
            attention_mask_ner = torch.unsqueeze(inputs['attention_mask'], 1)
            # [batch_size, 50, max_length], 复制50份
            attention_mask_ner = attention_mask_ner.expand(-1, len(self.label_map_seq.keys()), -1)
            # [batch_size * 50, max_length]
            attention_mask_ner_reshape = attention_mask_ner.reshape(batch_size * num_relations, max_length)
            # 选择预测正确的所有关系
            tmp1 = mask_output.unsqueeze(dim=1)  # [200, 1]
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
            input_ids_ner[rel_pos_mask.bool()] = relation_ids
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

            inputs_ner = {
                'input_ids': input_ids_ner,
                'token_type_ids': token_type_ids_ner,
                'attention_mask': attention_mask_ner_tmp,
            }

            try:
                outputs_ner = self.model_ner(**inputs_ner)[0]
            except BaseException:
                print('23')
            


            _, results = torch.max(outputs_ner, dim=2)
            results = results.cpu().tolist()
            results = [[self.label_map_ner[__] for __ in _] for _ in results]
            attention_position_np = rel_pos.cpu().numpy()

            attention_position_list = attention_position_np.tolist()
            predict_relation_list = relation_ids.long().tolist()
            input_ids_list = input_ids_ner.cpu().tolist()
                

            output = []
            input_ids = []
            for idx, result in enumerate(results):
                tmp1 = result[0: attention_position_list[idx]-1]
                tmp2 = input_ids_list[idx][0: attention_position_list[idx]-1]
                output.append(tmp1)
                input_ids.append(tmp2)
            
            input_split = torch.sum(mask_relation_output_sigmoid, dim=1)
            for i in range(1, batch_size):
                input_split[i] += input_split[i-1]
            tmp_input_ids = [input_ids[:input_split[0]]]
            tmp_output = [output[:input_split[0]]]
            for i in range(1, batch_size):
                tmp_input_ids.append(input_ids[input_split[i-1]:input_split[i]])
                tmp_output.append(output[input_split[i-1]:input_split[i]])
            output = tmp_output
            input_ids = tmp_input_ids

            # 将ner的句子转化为BIOES的标签之后把实体拿出来
            # processed_results_list_BIO = []
            # for result in processed_results_list:
            #     processed_results_list_BIO.append([self.label_map_ner[token] for token in result])


            # 把结果剥离出来
            index = 0
            triple_output = [[] for _ in range(batch_size)]

            # for each relation type or event type
            # by default, extract the first head and tail to construct the triples
            if self.mode == "triple":
                cnt = 0
                for ids_list, BIOS_list in zip(input_ids, output):
                    for ids, BIOS in zip(ids_list, BIOS_list):
                        labels = self.process(ids, BIOS)
                        # r = label_map_seq[predict_relation_list[index]]
                        r = predict_relation_list[index] - self.start_idx

                        if len(labels['subject']) == 0:
                            h = None
                        else:
                            h = labels['subject']
                            # h = ''.join(tokenizer.convert_ids_to_tokens(h))

                        if len(labels['object']) == 0:
                            t = None
                        else:
                            t = labels['object']
                            # t = ''.join(tokenizer.convert_ids_to_tokens(t))

                        # greedy select the head and tail
                        if h and t:
                            for hh in h:
                                for tt in t:
                                    triple_output[cnt].append([hh, r, tt])

                        index = index + 1
                    cnt += 1
            # 先不考虑
            # elif self.mode == "event":
            #     for ids, BIOS in zip(processed_input_ids_list, processed_results_list_BIO):
            #         triple_output.append(dict(event_type=predict_relation_list[index], argument=self.process(ids, BIOS)))

            return triple_output
        
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