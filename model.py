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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, batch_first=True):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm1 = nn.LSTM(embed_size, hidden_size, dropout=0.3, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, dropout=0.3, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size//2, hidden_size//4, dropout=0.3, batch_first=True)
        self.linear = nn.Linear(hidden_size//4, vocab_size)
    
    def forward(self, features, captions):
        captions_processed = self.embedding(captions)
        input_combined = torch.cat((features.unsqueeze(1), captions_processed), dim=1)        
        lstm_out1, _ = self.lstm1(input_combined)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out3, _ = self.lstm3(lstm_out2)
        output = self.linear(lstm_out3)
        output = output[:, :-1, :] #Given the last token of a caption, the model will predict a redundant token.
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        print(output)
        
        #print(inputs.shape)        
        #print(inputs)
        
        #None does not represent the blank state, which must be defined.
        states1 = (torch.zeros(1, inputs.size(0), self.lstm1.hidden_size).to(inputs.device),
                   torch.zeros(1, inputs.size(0), self.lstm1.hidden_size).to(inputs.device)) #Two states exist: hidden and cell states.
        states2 = (torch.zeros(1, inputs.size(0), self.lstm2.hidden_size).to(inputs.device),
                   torch.zeros(1, inputs.size(0), self.lstm2.hidden_size).to(inputs.device))
        states3 = (torch.zeros(1, inputs.size(0), self.lstm3.hidden_size).to(inputs.device),
                   torch.zeros(1, inputs.size(0), self.lstm3.hidden_size).to(inputs.device))       
        
        while True:
            if len(output) != max_len:
                lstm_test1, states1 = self.lstm1(inputs, states1)
                lstm_test2, states2 = self.lstm2(lstm_test1, states2)
                lstm_test3, states3 = self.lstm3(lstm_test2, states3)
                P_now = self.linear(lstm_test3.squeeze(1))
                _, tokenid_now = torch.max(P_now, dim=1)
            else:
                break

            output.append(tokenid_now.item())
            print(output)

            if tokenid_now.item() == 1:
                break
                
            inputs = self.embedding(tokenid_now)
            inputs = inputs.unsqueeze(1)
        
        return output