from transformers import BertForSequenceClassification, BertTokenizer
import torch
# Load model and tokenizer from saved folder
model = BertForSequenceClassification.from_pretrained("F:\\my_roberta_model")
tokenizer = BertTokenizer.from_pretrained("F:\\my_roberta_model")# or load from same folder if saved

# Use it
inputs = tokenizer("This is great!", return_tensors="pt")
outputs = model(**inputs)
labels = torch.tensor([1]).unsqueeze(0)
inputs = tokenizer('able move smacked head guy sitting front things got awkward', return_tensors="pt")
device = model.device
inputs = {key: value.to(device) for key, value in inputs.items()}

outputs = model(**inputs)
outlogits=outputs.logits 
probabilities = torch.sigmoid(outlogits)
emotions=['ANGER','FEAR','JOY','SADNESS','SURPRISE']
prob_to_emotions = zip(emotions, probabilities[0].tolist())
thisarray=list(prob_to_emotions)
resultfinal=sorted(thisarray, key=lambda x: x[1],reverse=True)
for i in resultfinal:
  if(i[1]>0.8):
    print(i[0])