import matplotlib.pyplot as plt

from nn import Linear, BatchNorm1d, Tanh
import torch.nn.functional as F

import torch
import argparse

import random
random.seed(42)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Input data in txt file")

    args = parser.parse_args()

    
    with open(args.data, 'r') as file:
        words = file.read().splitlines()

    # build the vocabulary 
    words = [''.join(filter(str.isalpha, word.lower())) for word in words]
    words = [word for word in words if word]  # Remove empty strings
    
    # Define the set of valid characters
    valid_chars = set('abcdefghijklmnopqrstuvwxyz.')
    
    # Create a list of valid characters present in the words
    all_chars = set(''.join(words))
    valid_chars.update(all_chars)

    chars = sorted(list(valid_chars))
    stoi = {s:i for i, s in enumerate(chars)}
    itos = {i:s for s, i in stoi.items()}
    vocab_size = len(itos)
    # Construct dataset
    block_size = 3

    def build_dataset(words):
        X, Y = [], []
        for w in words:
            context = [0] * block_size
            for ch in w + '.':
                ix = stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    Xtr,  Ytr  = build_dataset(words[:n1])    
    Xdev, Ydev = build_dataset(words[n1:n2])   
    Xte,  Yte  = build_dataset(words[n2:]) 
    n_embd = 10 
    n_hidden = 100 
    g = torch.Generator().manual_seed(2147483647) # for reproducibility

    C = torch.randn((vocab_size, n_embd), generator=g)
    layers = [
    Linear(n_embd * block_size, n_hidden, g, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, g, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, g, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, g, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, g, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size, g, bias=False), BatchNorm1d(vocab_size),
    ]

    with torch.no_grad():
    # last layer: make less confident
        layers[-1].gamma *= 0.1
    # all other layers: apply gain
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 1.0 # 5/3

    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    print(sum(p.nelement() for p in parameters)) 
    for p in parameters:
        p.requires_grad = True

    # same optimization as last time
    max_steps = 200000
    batch_size = 32
    lossi = []
    ud = []

    for i in range(max_steps):
    
    # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
        
        # forward pass
        emb = C[Xb] # embed the characters into vectors
        x = emb.view(emb.shape[0], -1) # concatenate the vectors
        for layer in layers:
            x = layer(x)
        loss = F.cross_entropy(x, Yb) # loss function
        
        # backward pass
        for layer in layers:
            layer.out.retain_grad() # AFTER_DEBUG: would take out retain_graph
        for p in parameters:
            p.grad = None
        loss.backward()
        
        # update
        lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
        for p in parameters:
            p.data += -lr * p.grad

        # track stats
        if i % 10000 == 0: # print every once in a while
            print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())
        with torch.no_grad():
            ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

        if i >= 1000:
            break # AFTER_DEBUG: would take out obviously to run full optimization
    

    # visualize histograms
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []
    for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
        if isinstance(layer, Tanh):
            t = layer.out
            print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__}')
    plt.legend(legends);
    plt.title('activation distribution')
    plt.show()

    # visualize histograms
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []
    for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
        if isinstance(layer, Tanh):
            t = layer.out.grad
            print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__}')
    plt.legend(legends);
    plt.title('gradient distribution')
    plt.show()

    # visualize histograms
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []
    for i,p in enumerate(parameters):
        t = p.grad
        if p.ndim == 2:
            print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'{i} {tuple(p.shape)}')
    plt.legend(legends)
    plt.title('weights gradient distribution');
    plt.show()

    plt.figure(figsize=(20, 4))
    legends = []
    for i,p in enumerate(parameters):
        if p.ndim == 2:
            plt.plot([ud[j][i] for j in range(len(ud))])
            legends.append('param %d' % i)
    plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
    plt.legend(legends);
    plt.show()