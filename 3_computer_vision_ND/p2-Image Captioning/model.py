import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # embedding layer transforms each word in a caption into a vector of desired size
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors as inputs and outputs hidden states
        self.lstm_layer = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # the linear layer that maps the hidden state the vocab_size
        self.linear_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        #We don't need to pass <stop> in the caption]
        captions = captions[:,:-1]
        embeds = self.embedding_layer(captions)

        # Output of CNN is features, which is the input to RNN
        inputs = torch.cat((features.unsqueeze(1),embeds), dim=1)
        lstm_out, _ = self.lstm_layer(inputs);   
        outputs = self.linear_layer(lstm_out); 
        return outputs 

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        out_sentence = []
        length = 0
        while (length != max_len):
            lstm_out, states = self.lstm_layer(inputs, states)
            lstm_out = lstm_out.squeeze(1)
            out = self.linear_layer(lstm_out)
            last = out.max(1)[1]
            #print (last, '\n', last.cpu().numpy()[0].item())
            out_sentence.append(last.item())
            if last.cpu().numpy()[0].item() == 1:
                break
            inputs = self.embedding_layer(last).unsqueeze(1)
            length += 1 
        return out_sentence   
    