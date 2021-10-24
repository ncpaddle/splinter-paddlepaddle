import paddle
import torch
import numpy as np

torch_model_path = "../splinter/init_params_t.bin"
torch_state_dict = torch.load(torch_model_path)

paddle_model_path = "../splinter/init_params_p.pdparams"
paddle_state_dict = {}

# State_dict's keys mapping: from torch to paddle
keys_dict = {
    # about embeddings
    "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
    "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",

    # about encoder layer
    'encoder.layer': 'encoder.layers',
    'attention.self.query': 'self_attn.q_proj',
    'attention.self.key': 'self_attn.k_proj',
    'attention.self.value': 'self_attn.v_proj',
    'attention.output.dense': 'self_attn.out_proj',
    'attention.output.LayerNorm': 'norm1',
    'intermediate.dense': 'linear1',
    'output.dense': 'linear2',
    'output.LayerNorm': 'norm2',

    # about cls predictions
    'cls.predictions.transform.dense': 'cls.predictions.transform',
    'cls.predictions.decoder.weight': 'cls.predictions.decoder_weight',
    'cls.predictions.transform.LayerNorm': 'cls.predictions.layer_norm',
    'cls.predictions.bias': 'cls.predictions.decoder_bias',
}

for torch_key in torch_state_dict:
    paddle_key = torch_key
    for k in keys_dict:
        if k in paddle_key:
            paddle_key = paddle_key.replace(k, keys_dict[k])
    if 'cls.span_predictions' not in paddle_key:
        if ('linear' in paddle_key) or ('proj' in  paddle_key) or ('vocab' in  paddle_key and 'weight' in  paddle_key) or ("dense.weight" in paddle_key) or ('transform.weight' in paddle_key) or ('seq_relationship.weight' in paddle_key):
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy().transpose())
        else:
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy())
    else:
        if 'dense.weight' in paddle_key:
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy().transpose())
        else:
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy())

    print("torch: ", torch_key,"\t", torch_state_dict[torch_key].shape)
    print("paddle: ", paddle_key, "\t", paddle_state_dict[paddle_key].shape, "\n")

paddle.save(paddle_state_dict, paddle_model_path)

print(type(paddle_state_dict))


# paddle_params = paddle.load('../splinter/model_state.pdparams')
#
# for key in paddle_params.keys():
#     print(key)