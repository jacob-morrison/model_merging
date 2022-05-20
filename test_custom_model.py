import os
os.environ['TRANSFORMERS_CACHE'] = '/home/acd13578qu/data/.cache/huggingface'

from transformers import RobertaConfig, RobertaModel, ViTFeatureExtractor, ViTModel, TFAutoModelForSequenceClassification, AutoTokenizer, TFViTModel
from pprint import pprint


# TODO: Need to figure out how to view all the Tf layers
# This will let me see if they're in the right order/etc

configuration = RobertaConfig()
roberta_model = TFAutoModelForSequenceClassification.from_pretrained(
    # '/home/acd13578qu/scratch/roberta_actual/checkpoints/checkpoint_best.pt',
    'textattack/roberta-base-RTE',
    from_pt=True)
roberta_layers = roberta_model.layers[0]
vit_model = TFViTModel.from_pretrained('google/vit-base-patch16-224-in21k', from_pt=True)
vit_layers = vit_model.layers[0]

print('roberta layers:')
print(roberta_layers.encoder.summary())
print(len(roberta_layers.encoder.layer))
print(vit_layers.encoder.summary())
print(len(vit_layers.encoder.layer))
# pprint(dir(roberta_model))
# pprint(dir(roberta_layers))
# print(roberta_model.summary())
# print('vit layers:')
# print(vit_layers)

# # print('roberta:')
# roberta_params = []
# roberta_shapes = []
# roberta_total_params = 0
# for name, param in roberta_model.named_parameters():
#     if param.requires_grad:
#         # print(str(name))
#         start = 1
#         roberta_shapes.append(param.data.size())
#         for elem in list(param.data.size()):
#             start *= elem
#         roberta_total_params += start
#         roberta_params.append(name)

# # print('vit:')
# vit_params = []
# vit_shapes = []
# vit_total_params = 0
# for name, param in vit_model.named_parameters():
#     if param.requires_grad:
#         # print(str(name))
#         start = 1
#         vit_shapes.append(param.data.size())
#         for elem in list(param.data.size()):
#             start *= elem
#         vit_total_params += start
#         vit_params.append(name)

# for i in range(5, len(roberta_params)):
#     if roberta_params[i] == vit_params[i-1]:
#         print(roberta_params[i])
#         print(roberta_shapes[i])
#         print(vit_params[i-1])
#         print(vit_shapes[i-1])
#         print()

# for bert_param, vit_param, bert_shape, vit_shape in zip(bert_params, vit_params, bert_shapes, vit_shapes):
#     if bert_shape != vit_shape:
#         print('Mismatch!!')
#         print(bert_shape)
#         print(vit_shape)
#         print(bert_param)
#         print(vit_param)
#         print()