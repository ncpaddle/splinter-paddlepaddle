import paddle
import paddle.nn as nn
from paddlenlp.transformers import BertPretrainedModel, BertModel, RobertaModel, BertForQuestionAnswering, PretrainedModel
import numpy as np
from paddlenlp.transformers import PretrainedTokenizer

def torch_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:  # 最后一维
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            dim_index = paddle.expand(paddle.reshape(paddle.arange(x.shape[k], dtype=index.dtype), reshape_shape),
                                      index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0])
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out

def gather_positions(input_tensor, positions):
    """
    :param input_tensor: shape [batch_size, seq_length, dim]
    :param positions: shape [batch_size, num_positions]
    :return: [batch_size, num_positions, dim]
    """
    _, seq_length, dim = input_tensor.shape
    bs, nump = positions.shape
    index = paddle.expand(positions.unsqueeze(-1), shape=[bs, nump, dim])  # [batch_size, num_positions, dim]
    gathered_output = torch_gather(input_tensor, dim=1, index=index)  # [batch_size, num_positions, dim]
    return gathered_output


class ModelWithQASSHead(BertPretrainedModel):
    def __init__(self, bert, replace_mask_with_question_token=True,
                 mask_id=103, question_token_id=104, sep_id=102, initialize_new_qass=False):
        super(BertPretrainedModel, self).__init__()
        self.bert = bert
        self.initialize_new_qass = initialize_new_qass
        self.cls = ClassificationHead() if not self.initialize_new_qass else None
        self.new_cls = ClassificationHead() if self.initialize_new_qass else None

        self.sep_id = sep_id
        self.mask_id = mask_id
        self.question_token_id = question_token_id
        self.replace_mask_with_question_token = replace_mask_with_question_token


    def get_cls(self):
        if self.initialize_new_qass:
            return self.new_cls
        return self.cls

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                masked_positions=None, start_positions=None, end_positions=None):
        if attention_mask is not None:
            flag = paddle.to_tensor(input_ids == self.sep_id, dtype=attention_mask.dtype)
            y = paddle.zeros_like(attention_mask)
            attention_mask = paddle.where(flag > 0, y, attention_mask)

        if self.replace_mask_with_question_token:
            flag2 = paddle.to_tensor(input_ids == self.mask_id, dtype=input_ids.dtype)
            y2 = paddle.ones_like(input_ids, dtype=input_ids.dtype) * self.question_token_id
            input_ids = paddle.where(flag2 > 0, y2, input_ids)

        mask_positions_were_none = False
        if masked_positions is None:
            temp = input_ids == self.question_token_id
            ttt = paddle.to_tensor(temp, dtype=paddle.int32)
            masked_position_for_each_example = paddle.argmax(ttt, axis=-1)
            masked_positions = masked_position_for_each_example.unsqueeze(-1)
            mask_positions_were_none = True

        attention_mask_expand = paddle.unsqueeze(attention_mask, axis=[1, 2])
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask_expand, token_type_ids=token_type_ids)
        sequence_output = outputs[0]  # [batch_size, max_length, dim]
        return sequence_output  # 1check_forward中align_pretrainedModel.py中使用

        cls = self.get_cls()
        start_logits, end_logits = cls(sequence_output, masked_positions)

        if mask_positions_were_none:
            start_logits, end_logits = start_logits.squeeze(1), end_logits.squeeze(1)

        if attention_mask is not None:
            start_logits = start_logits + (1 - attention_mask) * -10000.0
            end_logits = end_logits + (1 - attention_mask) * -10000.0
        outputs = outputs[2:]
        outputs = (start_logits, end_logits,) + outputs

        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.shape[1]
            start_positions = paddle.clip(start_positions, 0, ignored_index)
            end_positions = paddle.clip(end_positions, 0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2
            print('total_loss', total_loss)
            print('start_loss', start_loss)
            print('end_loss', end_loss)
            outputs = (total_loss,) + outputs

        return outputs


class FullyConnectedLayer(nn.Layer):
    def __init__(self, input_dim, output_dim, hidden_act="gelu"):
        super(FullyConnectedLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense = nn.Linear(in_features=self.input_dim, out_features=self.output_dim,
                               weight_attr=nn.initializer.KaimingUniform(), bias_attr=nn.initializer.KaimingUniform())
        self.act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(self.output_dim)

    def forward(self, inputs):
        temp = self.dense(inputs)
        temp = self.act_fn(temp)
        temp = self.LayerNorm(temp)
        return temp


class QuestionAwareSpanSelectionHead(nn.Layer):
    def __init__(self, hidden_size=768):
        super(QuestionAwareSpanSelectionHead, self).__init__()
        self.query_start_transform = FullyConnectedLayer(hidden_size, hidden_size)
        self.query_end_transform = FullyConnectedLayer(hidden_size, hidden_size)
        self.start_transform = FullyConnectedLayer(hidden_size, hidden_size)
        self.end_transform = FullyConnectedLayer(hidden_size, hidden_size)
        self.start_classifier = paddle.create_parameter([hidden_size, hidden_size], dtype=paddle.float32)
        self.end_classifier = paddle.create_parameter([hidden_size, hidden_size], dtype=paddle.float32)

    def forward(self, inputs, positions):
        # return inputs # diff: 5.722e-06
        gathered_reps = gather_positions(inputs, positions)  # diff: 3e-07
        query_start_reps = self.query_start_transform(gathered_reps)  # diff: 1.34e-07
        query_end_reps = self.query_end_transform(gathered_reps)
        start_reps = self.start_transform(inputs)  # diff: 2.61e-07
        end_reps = self.end_transform(inputs)

        temp = paddle.matmul(query_start_reps, self.start_classifier)  # diff: 3.69778e-07
        start_reps = paddle.transpose(start_reps, perm=[0, 2, 1]) # diff: 2.612227660847566e-07
        start_logits = paddle.matmul(temp, start_reps) # diff: 0.0012255377369001508

        temp = paddle.matmul(query_end_reps, self.end_classifier) # diff: 3.106023598320462e-07
        end_reps = paddle.transpose(end_reps, perm=[0, 2, 1]) # diff: 2.669026457624568e-07
        end_logits = paddle.matmul(temp, end_reps) # diff: 0.0008031659526750445

        return start_logits, end_logits


class ClassificationHead(nn.Layer):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        self.span_predictions = QuestionAwareSpanSelectionHead()

    def forward(self, inputs, positions):
        return self.span_predictions(inputs, positions)



