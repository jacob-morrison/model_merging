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

def convert_label(label):
    if label == 0:
        return 'ENTAILMENT'
    elif label == 1:
        return 'NEUTRAL'
    elif label == 2:
        return 'CONTRADICTION'
    else:
        return 'WTF'

# tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
# model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
tokenizer = AutoTokenizer.from_pretrained("ishan/bert-base-uncased-mnli")
model = AutoModelForSequenceClassification.from_pretrained("ishan/bert-base-uncased-mnli")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

inputs = []
labels = []
dataset = load_dataset('glue', 'mnli_matched')
for i in range(len(dataset['validation'])):
# for i in range(10):
    row = dataset['validation'][i]
    labels.append(convert_label(int(row['label'])))
    inputs.append(row['premise'] + ' </s></s> ' + row['hypothesis'])

results = classifier(inputs)
print(labels)
print(results)
correct_count = 0
for i in range(len(labels)):
    if labels[i] == results[i]['label']:
        correct_count += 1
print(1. * correct_count / len(labels))