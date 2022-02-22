# from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
# import torch
# from datasets import load_dataset

# model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli', num_labels=3)
# tokenizer = RobertaTokenizer.from_pretrained("roberta-large-mnli") # check if this is the right tokenizer
# dataset = load_dataset('glue', 'mnli_matched')
# print(dataset['validation'][0])
# for premise, hypothesis, label, idx in dataset['validation']:
#     inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
#     labels = torch.tensor([label]).unsqueeze(0)  # Batch size 1
#     outputs = model(**inputs, labels=labels)
#     loss = outputs.loss
#     logits = outputs.logits
#     break

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

inputs = []
labels = []
dataset = load_dataset('glue', 'mnli_matched')
print(dataset['validation'][0])
for i in range(2):
    print(i)
    row = dataset['validation'][i]
    print(row)
    labels.append(int(row['label']))
    inputs.append(row['premise'] + ' </s></s> ' + row['hypothesis'])
    break

results = classifier(inputs)
print(labels)
print(results)