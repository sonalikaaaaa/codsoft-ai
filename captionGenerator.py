import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
from transformers import BertTokenizer
import numpy as np

#Load pre-trained ResNet model
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

#Remove the final classification layer
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(ResNetFeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = x.view(x.size(0), -1)
        return x
    
feature_extractor = ResNetFeatureExtractor(resnet_model)

#Define image transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
])

class CaptioningModel(nn.Module):
    def __init__(self, feature_dim, vocab_size, embed_size, hidden_size):
        super(CaptioningModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + feature_dim, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        input = torch.cat((features.unsqueeze(1), embeddings), dim = 1)
        outputs, _ = self.lstm(inputs)
        logits = self.fc(outputs)
        return logits

# Dummy tokenizer
class DummyTokenizer:
    def __init__(self):
        self.vocab = {'<start>': 0, '<end>': 1, 'a': 2, 'cat': 3, 'on': 4, 'the': 5, 'mat': 6}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        return [self.vocab.get(word, 0) for word in text.split()]
    
    def decode(self, indices):
        return ' '.join([self.inv_vocab.get(i, '<unk>') for i in indices])
    
    def __len__(self):
        return len(self.vocab)

tokenizer = DummyTokenizer()
vocab_size = len(tokenizer)

# Initialize captioning model
embed_size = 256
hidden_size = 512
feature_dim = 2048
captioning_model = CaptioningModel(feature_dim, vocab_size, embed_size, hidden_size)
captioning_model.eval()

def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = feature_extractor(image)
    return features

def generate_caption(model, feature_vector, tokenizer, max_length=20):
    model.eval()
    with torch.no_grad():
        caption = []
        input_seq = torch.tensor([[tokenizer.vocab['<start>']]]).long()
        for _ in range(max_length):
            outputs = model(feature_vector, input_seq)
            _, predicted = outputs[:, -1].max(1)
            word = tokenizer.decode(predicted.tolist())
            caption.append(word)
            if word == '<end>':
                break
            input_seq = torch.cat((input_seq, predicted.unsqueeze(0)), dim=1)
        return ' '.join(caption)

# Example usage
image_path = 'example.jpg'  # Replace with your image path
features = extract_features(image_path)
caption = generate_caption(captioning_model, features, tokenizer)
print("Generated Caption:", caption)
