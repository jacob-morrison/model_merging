from transformers import RobertaTokenizer, RobertaModel, ViTFeatureExtractor, ViTModel

test_model = RobertaModel.from_pretrained('/home/acd13578qu/scratch/roberta_actual/checkpoints/checkpoint_best.pt', from_pt=True)

bert_params = []
bert_shapes = []
bert_total_params = 0
for name, param in test_model.named_parameters():
    if param.requires_grad:
        print(str(name))
        start = 1
        bert_shapes.append(param.data.size())
        for elem in list(param.data.size()):
            start *= elem
        bert_total_params += start
        bert_params.append(name)