import argparse
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Input data in txt format")
    args = parser.parse_args()
    
    with open(args.data, 'r') as file:
        words = file.read().splitlines()
    
    # Filter out non-alphabetic characters and convert to lowercase
    words = [''.join(filter(str.isalpha, word.lower())) for word in words]
    words = [word for word in words if word]  # Remove empty strings
    
    # Define the set of valid characters
    valid_chars = set('abcdefghijklmnopqrstuvwxyz.')
    
    # Create a list of valid characters present in the words
    all_chars = set(''.join(words))
    valid_chars.update(all_chars)
    
    N = torch.zeros((len(valid_chars), len(valid_chars)), dtype=torch.int32)
    
    chars = sorted(list(valid_chars))
    stoi = {s:i for i, s in enumerate(chars)}
    itos = {i:s for s, i in stoi.items()}

    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            if ch1 in valid_chars and ch2 in valid_chars:
                ix1 = stoi[ch1]
                ix2 = stoi[ch2]
                N[ix1, ix2] += 1
                
    ##### FOR VISUALIZING THE BIGRAM MATRIX####
    # plt.figure(figsize=(36, 36))  
    # plt.imshow(N, cmap='Blues')  

    # for i in range(27):
    #     for j in range(27):
    #         chstr = itos[i] + itos[j] 
    #         plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
    #         plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')

    # plt.axis('off')  
    # plt.show()
    
    P = (N+1).float() # Adding 1 prevents infinte value
    P /= P.sum(1, keepdims=True)

    g = torch.Generator().manual_seed(23234234243)
    for i in range(5):
        out = []
        ix = 0
        while True:
            p = P[ix]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))
    
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1
    print(f'{log_likelihood=}')
    nll = -log_likelihood
    print(f'{nll=}')
    print(f'{nll/n}')

    # Creating training set of bigrams (x, y)
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num = xs.nelement()

    W = torch.randn(len(valid_chars), len(valid_chars), generator=g, requires_grad=True)
    for i in range(100):
        # forward pass
        xenc = F.one_hot(xs, num_classes=len(valid_chars)).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True) # probability for next character
        loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()

        # backward pass
        W.grad = None
        loss.backward()

        # update
        W.data += -50 * W.grad
    print("loss:",loss.item())

    # Sampling from our neural net model
    for i in range(5):
        out = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=len(valid_chars)).float()
            logits = xenc @ W
            counts = logits.exp()
            p = counts / counts.sum(1, keepdims=True)

            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))
