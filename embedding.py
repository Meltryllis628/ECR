import pandas as pd
import torch
from torch.autograd import Variable
import re
import os

train = pd.read_csv("train_tokenized.csv")
print("Read file")
small = train.loc[0:34877]

def text_to_train(text, context_window, w2i):
    
    data = []

    for i in range(context_window, len(text) - context_window):
        context = [text[i+e] for e in range(-context_window, context_window+1) if i+e != i]
        target = text[i]
        data.append((context, target))
        
    return data

def generate_data(train, context_window):
    
    vocab = set()
    for i in train.index:
        raw = train.loc[i, "Tokenization"]
        sentance = re.sub('[\[\]\s\']','',raw).split(",")
        for word in sentance:
            vocab.add(word)
    
    word2index = {w:i for i,w in enumerate(vocab)}
    index2word = {i:w for i,w in enumerate(vocab)}
    
    data = []
    for i in train.index:
        raw = train.loc[i, "Tokenization"]
        data += text_to_train(re.sub('[\[\]\s\']','',raw).split(","), context_window, word2index)

    return vocab, data, word2index, index2word

def words_to_tensor(words: list, w2i: dict, dtype = torch.FloatTensor):
    tensor =  dtype([w2i[word] for word in words])
    tensor = tensor.to("cuda")
    return Variable(tensor)

vocab, data, word2index, index2word = generate_data(small,2)
print("Generate Data 11")
print("Vocab num: {}, Data num: {}".format(len(vocab), len(data)))

class CBOW(torch.nn.Module):

    def __init__(self, context_size, embedding_size, vocab_size):
        super(CBOW, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_size)
        self.linear = torch.nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        lookup_embeds = self.embeddings(inputs)
        embeds = lookup_embeds.sum(dim=0)
        out = self.linear(embeds)
        #out = F.log_softmax(out, dim = -1)
        return out
    
model = CBOW(context_size = 2, embedding_size = 300, vocab_size = len(vocab))
model = model.to("cuda")
model.train()

learning_rate = 0.001
epochs = 1000
torch.manual_seed(19260817)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def get_prediction(context, model, word2index, index2word):
    
    # Get into eval() mode
    model.eval()
    ids = words_to_tensor(context, word2index, dtype = torch.LongTensor)
    
    # Forward pass
    prediction = model(ids)
    # Reshape to cover for absence of minibatches (needed for loss function)
    prediction = torch.reshape(prediction, (1, len(vocab)))
    _, index = torch.max(prediction, 1)
    
    return index2word[index.item()]

def check_accuracy(model, data, word2index, index2word):
    
    # Compute accuracy
    correct = 0
    for context, target in data:
        prediction = get_prediction(context, model, word2index, index2word)
        if prediction == target:
            correct += 1
            
    return correct/len(data)

losses = []
accuracies = []

def load_model(save_name, optimizer, model):
    model_data = torch.load(save_name)
    model.load_state_dict(model_data['model_state_dict'])
    optimizer.load_state_dict(model_data['optimizer_state_dict'])
    model.train()
    print("model {} load success".format(model_data['epoch']))
    return(model_data['epoch']+1) 

def search_for_model():
    g = os.walk(".")  
    pattern = r"pt(\d+)\.tar"
    max_step = 0
    for path,dir_list,file_list in g:  
        for file_name in file_list:  
            match = re.search(pattern, file_name)
            if match:
                number = int(match.group(1))
                if number > max_step:
                    max_step = number
    if max_step>0:
        return "pt{}.tar".format(max_step)
    else:
        return "null"

checkpoint = search_for_model()
if (checkpoint != "null"):
    existEpoch = load_model(search_for_model(),optimizer,model)


for epoch in range(existEpoch,epochs):
    total_loss = 0
    for context, target in data:
        
        # Prepare data
        ids = words_to_tensor(context, word2index, dtype = torch.LongTensor)
        target = words_to_tensor([target], word2index, dtype = torch.LongTensor)
        
        # Forward pass
        model.zero_grad()
        output = model(ids)
        # Reshape to cover for absence of minibatches (needed for loss function)
        output = torch.reshape(output, (1, len(vocab)))
        loss = loss_func(output, target)
        
        # Backward pass and optim
        loss.backward()
        optimizer.step()
        
        # Loss update
        total_loss += loss.data.item()
    
    # Display
    accuracy = check_accuracy(model, data, word2index, index2word)
    print("Accuracy after epoch {} is {}".format(epoch, accuracy))
    accuracies.append(accuracy)
    losses.append(total_loss)
    outfile = open("output",'a',encoding="utf-8")
    print("{},{},{}".format(epoch,total_loss,accuracy),file=outfile)

    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, "pt{}.tar".format(epoch+1))




