from transformers import RobertaTokenizer, RobertaModel, ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
from torchsummary import summary
import torch
from torch.utils.tensorboard import SummaryWriter

roberta_mnli_model = RobertaModel.from_pretrained('roberta-large-mnli')
roberta_xlm_model = RobertaModel.from_pretrained('xlm-roberta-large')


# writer=SummaryWriter('content/logsdir')
# roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# roberta_model = RobertaModel.from_pretrained('roberta-base')

# roberta_layer = roberta_model.encoder.layer[0]
# print(roberta_layer)

# summary(roberta_layer, input_size=(1,768))
# dummy_input = torch.rand(1, 10, 768)
# writer.add_graph(roberta_layer, dummy_input)
# writer.close()

# inputs = roberta_tokenizer("Hello, my cat is very cute", return_tensors="pt")
# print(inputs.input_ids.size())

# summary(roberta_model, input_size=(5,))

# roberta_params = set()
# print("RoBERTa params:")
# for name, param in roberta_model.named_parameters():
#     if param.requires_grad:
#         print(str(name))
#         print(param.data.size())
#         roberta_params.add(name)

# vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# inputs = feature_extractor(images=image, return_tensors="pt")
# print(inputs)

# vit_layer = vit_model.encoder.layer[0]
# writer.add_graph(vit_layer, dummy_input)
# writer.close()



# print(vit_layer)
# summary(vit_layer, input_size=(1,768))

bert_params = []
bert_shapes = []
bert_total_params = 0
for name, param in bert.named_parameters():
    if param.requires_grad:
        print(str(name))
        start = 1
        bert_shapes.append(param.data.size())
        for elem in list(param.data.size()):
            start *= elem
        bert_total_params += start
        bert_params.append(name)

roberta_params = []
roberta_shapes = []
roberta_total_params = 0
for name, param in roberta.named_parameters():
    if param.requires_grad:
        print(str(name))
        start = 1
        roberta_shapes.append(param.data.size())
        for elem in list(param.data.size()):
            start *= elem
        roberta_total_params += start
        roberta_params.append(name)

for bert_param, roberta_param, bert_shape, roberta_shape in zip(bert_params, roberta_params, bert_shapes, roberta_shapes):
    if bert_shape != roberta_shape:
        print('Mismatch!!')
        print(bert_shape)
        print(bert_param)
        print(roberta_shape)
        print(roberta_param)




# print()
# for param in sorted(vit_params):
#     if param not in roberta_params:
#         print(param + " not in roberta")
# print()
# for param in sorted(roberta_params):
#     if param not in vit_params:
#         print(param + " not in vit")

# print(roberta_model)
# inputs = roberta_tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = roberta_model(**inputs)

print("done")