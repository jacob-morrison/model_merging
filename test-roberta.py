from transformers import RobertaTokenizer, RobertaModel, ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
from torchsummary import summary
import torch
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter('content/logsdir')

# roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

roberta_layer = roberta_model.encoder.layer[0]
print(roberta_layer)

# summary(roberta_layer, input_size=(1,768))
dummy_input = torch.rand(1, 10, 768)
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

vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
inputs = feature_extractor(images=image, return_tensors="pt")
# print(inputs)

# vit_layer = vit_model.encoder.layer[0]
# writer.add_graph(vit_layer, dummy_input)
# writer.close()



# print(vit_layer)
# summary(vit_layer, input_size=(1,768))

# vit_params = set()
# print("ViT params:")
# for name, param in vit_model.named_parameters():
#     if param.requires_grad:
#         print(str(name))
#         print(param.data.size())
#         vit_params.add(name)




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