from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import os
import torch
import random
import pickle
from models.lstm.lstm_model import Model

# constants
device = torch.device('cpu')
max_seq_len = 30
temperature = 0.6

# paths
script_dir = os.path.dirname(__file__)
model_path = os.path.join(
    script_dir, '../models/lstm/best-validation-lstm-lm-weights.pth')
vocab_path = os.path.join(script_dir, '../models/lstm/lstm-vocab.pkl')
tokenizer_path = os.path.join(script_dir, '../models/lstm/lstm-tokenizer.pkl')

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(vocab)
embedding_dim = 1024
hidden_dim = 1024
num_layers = 2
dropout_rate = 0.65
tie_weights = True

model = Model(vocab_size, embedding_dim, hidden_dim,
              num_layers, dropout_rate, tie_weights).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))


def infer(request):

    # tokens = list(vocab.get_itos())
    tokens = ['Who', 'What', 'When', 'Why', 'How',
              'The', 'But', 'Since', 'A', 'An', 'This', 'It', 'That', 'There', 'They', 'We', 'You', 'My', 'Their', 'Our', 'And', 'Or']
    init_prompt = '<eos> ' + (random.choice(tokens))

    model.eval()
    tokens = tokenizer(init_prompt)
    indices = [vocab[token] for token in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probabilities = torch.softmax(
                prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probabilities, num_samples=1).item()

            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(
                    probabilities, num_samples=1).item()
            if prediction == vocab['<eos>']:
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[idx] for idx in indices]
    brainrot = ' '.join(tokens[1:])

    return JsonResponse({'data': brainrot})
