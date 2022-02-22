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
from pprint import pprint
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, Trainer

def convert_label(label):
    if label == 0:
        return 'ENTAILMENT'
    elif label == 1:
        return 'NEUTRAL'
    elif label == 2:
        return 'CONTRADICTION'
    else:
        return 'WTF'

def convert_label_bert(label):
    if label == 0:
        return 'LABEL_0'
        # return 'LABEL_2'
    elif label == 1:
        return 'LABEL_1'
        # return 'LABEL_0'
    elif label == 2:
        return 'LABEL_2'
        # return 'LABEL_1'
    else:
        return 'WTF'

MODEL = 'BERT'

if MODEL == 'RoBERTa':
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
elif MODEL == 'BERT':
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MNLI")
    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

inputs = []
labels = []
dataset = load_dataset('glue', 'mnli', split='validation_matched[:100]')
for i in range(len(dataset)):
# for i in range(100):
    row = dataset[i]
    if MODEL == 'RoBERTa':
        labels.append(convert_label(int(row['label'])))
    elif MODEL == 'BERT':
        labels.append(convert_label_bert(int(row['label'])))

    if MODEL == 'RoBERTa':
        inputs.append(row['premise'] + ' ' + row['hypothesis']) # TODO: Should I include sep tokens?
    elif MODEL == 'BERT':
        inputs.append(row['premise'] + ' ' + row['hypothesis']) # Doesn't work
    
prediction_counts = {}
results = classifier(inputs)
correct_count = 0
for i in range(len(results)):
    results[i]['prediction'] = labels[i]
    if results[i]['label'] not in prediction_counts:
        prediction_counts[results[i]['label']] = {}
    if results[i]['prediction'] not in prediction_counts[results[i]['label']]:
        prediction_counts[results[i]['label']][results[i]['prediction']] = 0
    prediction_counts[results[i]['label']][results[i]['prediction']] += 1 
    if results[i]['label'] == results[i]['prediction']:
        correct_count += 1

pprint(prediction_counts)

pprint(results)
print(1. * correct_count / len(labels))

# tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
# model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

# def preprocess_function(examples):
#     return tokenizer(examples["text"], truncation=True)

# tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MNLI")
# model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI")
# classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# inputs = []
# labels = []
# dataset = load_dataset('glue', 'mnli_matched', split='validation[:32]')

# tokenized_data = dataset.map(preprocess_function, batched=True)

# test_trainer = Trainer(model) 
# raw_pred, _, _ = test_trainer.predict(dataset)
# print(raw_pred)

# for i in range(len(dataset['validation'])):
# for i in range(10):
#     row = dataset['validation'][i]
#     # labels.append(convert_label_bert(int(row['label'])))
#     labels.append(int(row['label']))
#     inputs.append((row['premise'], row['hypothesis']))

# correct_count = 0
# for i in range(len(inputs)):
#     gold_label = labels[i]
#     premise, hypothesis = inputs[i]
#     # Try with and without tuples?
#     encoded_text = tokenizer.encode_plus(premise, hypothesis, return_tensors='pt')
#     outputs = model(**encoded_text)
#     # print(outputs.logits)
#     print(gold_label)
#     print(torch.argmax(outputs.logits).item())
#     print()
#     if gold_label == torch.argmax(outputs.logits).item():
#         correct_count += 1

# print(1. * correct_count / len(labels))


# results = classifier(inputs)
# print(labels)
# print(results)
# correct_count = 0
# for i in range(len(labels)):
#     if labels[i] == results[i]['label']:
#         correct_count += 1
# print(1. * correct_count / len(labels))