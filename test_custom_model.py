import os
os.environ['TRANSFORMERS_CACHE'] = '/home/acd13578qu/data/.cache/huggingface'

from transformers import RobertaConfig, RobertaModel, ViTFeatureExtractor, ViTModel
from pprint import pprint

configuration = RobertaConfig()
roberta_model = RobertaModel.from_pretrained(
    '/home/acd13578qu/scratch/roberta_actual/checkpoints/checkpoint_best.pt',
    config=configuration)
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

print('roberta:')
bert_params = []
bert_shapes = []
bert_total_params = 0
for name, param in roberta_model.named_parameters():
    if param.requires_grad:
        print(str(name))
        start = 1
        bert_shapes.append(param.data.size())
        for elem in list(param.data.size()):
            start *= elem
        bert_total_params += start
        bert_params.append(name)

pprint(zip(bert_params, bert_shapes))

print('vit:')
roberta_params = []
roberta_shapes = []
roberta_total_params = 0
for name, param in vit_model.named_parameters():
    if param.requires_grad:
        print(str(name))
        start = 1
        roberta_shapes.append(param.data.size())
        for elem in list(param.data.size()):
            start *= elem
        roberta_total_params += start
        roberta_params.append(name)

pprint(zip(roberta_params, roberta_shapes))

# for bert_param, roberta_param, bert_shape, roberta_shape in zip(bert_params, roberta_params, bert_shapes, roberta_shapes):
#     if bert_shape != roberta_shape:
#         print('Mismatch!!')
#         print(bert_shape)
#         print(roberta_shape)
#         print(bert_param)
#         print(roberta_param)
#         print()