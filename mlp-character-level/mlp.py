import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

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
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])

    g = torch.Generator().manual_seed(2147483647)
    # From the bengio paper
    ## The C represents the embedding lookup table
    # We have only 27 valid characters so using 27
    ## Let's further make it more general by using len(valid_chars)
    C = torch.randn((len(valid_chars), 10), generator=g)
    # Now since we have 10 dimensional embedding for each words
    ## So the tanh layer receives 3 *10 =30 inputs  
    W1 = torch.randn((30, 300), generator=g)
    b1 = torch.randn(300, generator=g)
    W2 = torch.randn((300, len(valid_chars)), generator=g)
    b2 = torch.randn(len(valid_chars), generator=g)
    parameters = [C, W1, b1, W2, b2]

    # enabling grad calcualation during forward pass
    for p in parameters:
        p.requires_grad = True
    
    # Checking the learning rates from 0.001 to 1
    lre = torch.linspace(-3, 0, 1000)
    lrs = 10**lre

    lri = []
    lossi = []
    stepi = []

    for i in range(200000):
        # Minibatch of size 32
        ix = torch.randint(0, Xtr.shape[0], (32,))

        # Forward pass
        # (32, 3, 10)
        emb = C[Xtr[ix]]
        h = torch.tanh(emb.view(-1, 30) @ W1 + b1) #(32, 300)
        logits = h @ W2 + b2 #(32, 27)
        loss = F.cross_entropy(logits, Ytr[ix])

        # Backward pass
        for p in parameters:
            p.grad = None
        
        loss.backward()

        # Optimization
        # lr = lrs[i]
        lr = 0.1 if i < 100000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

        stepi.append(i)
        lossi.append(loss.log10().item())

    print(loss.item())
    plt.plot(stepi, lossi)

    # Calculating loss on train set
    emb = C[Xtr]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr)
    print("train_loss", loss.item())

    # Calculating loss in validation set
    emb = C[Xdev]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    dev_loss = F.cross_entropy(logits, Ydev)
    print("valid_loss", dev_loss.item())

    # visualizing dimensions 0 and 1 of the embedding matrix C
    plt.figure(figsize=(6, 6))
    plt.scatter(C[:,0].data, C[:, 1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color='white')
    plt.grid('minor')
    plt.show()

    # sample from the model
    g = torch.Generator().manual_seed(2147483647 + 10)

    for _ in range(20):
        
        out = []
        context = [0] * block_size # initialize with all ...
        while True:
            emb = C[torch.tensor([context])] # (1,block_size,d)
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
            
        print(''.join(itos[i] for i in out))