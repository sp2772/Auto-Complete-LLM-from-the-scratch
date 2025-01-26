# %%
print('boo')


# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import math

# %%
#import argparse

#parser = argparse.ArgumentParser(description="cmd arguments!")
#parser.add_argument('-batch_size',type = str, required= True,help='Please proved a batch size')

#args = parser.parse_args()
#print(f'batch_size: {args.batch_size}')


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


#batch_size =32 #args.batch_size
max_num_of_dumps=3
block_size=128
max_iters=100
learning_rate= 3e-4    #, 1e-3,1e-4
eval_iters =100
n_embd = 384    #number of total dimesions we want to capture from all the heads concatenated together
n_layer =8  #number of decoder blocks
n_head= 8   #number of heads 
dropout= 0.15  #20% neuron dropout
batch_size = 64 if max_iters < 10000 else 128


# %%
chars=""
with open("vocab.txt",'r',encoding='utf-8') as f:
    text= f.read()
    chars = sorted(list(set(text)))



vocab_size=len(chars)


# %%
stoi = {ch : i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode= lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


# %%
 


# %%

# %%
class Head(nn.Module):
    """ one Head of self-attention that applies scaled-dot-product attention """

    def __init__(self, head_size):
        super().__init__()
        
        #transform n_embd to head_size without bias (384 to 96 features)
        self.key= nn.Linear(n_embd, head_size, bias=False)
        self.query= nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd,head_size, bias =False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))     #register the no-look ahead masking to the model state (cachng for faster computations and prevent overhead comnputations)
        

        self.dropout = nn.Dropout(dropout)             #follow neuron droput procedure to reduce overfitting

    def forward(self, x):
        #input of size (batch, time-step, channels) (B,T,C)
        #output of size (batch, time-step, head_size)

        B,T,C = x.shape            #we saw how to unpack the size in torch fn example
        k= self.key(x)              #(B,T,hs)  key, query are outputs of the linear transformation layer with output layer dimension: (B, T, head_size)
        q= self.query(x)            #(B,T,hs)

        #compute attention scores ("affinities")
        #attention weights 'wei'  We do transpose to  make the operand matrices in a table form (flip the 2nd last dimension with the last dimension of key so that the query can be matrix multipled with the key)
        #And then do a 1/sqrt(length of a row of keys/queries) scaling on the weights
        #Compare scaling with a real life example:
        #(while hearing people inside a room: to control the loudness of each feature/head in the same room, to weight everything a bit evenly even when somebody(head) is too loud or too quiet)
        
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5                   #(B,T,hs) @ (B,T,hs) -> (B, T, T)      #the dimension of 'key' flipped and result is scaled

        #mask file: expose T one by one (thats why its called Time-step) 
        #[1, 0, 0] -> [ 1, -inf , -inf]
        #[1, 0.6, 0] -> [ 1, 0.6 , -inf]
        #[1, 0.6, 0.4]  
        wei = wei.masked_fill(self.tril[:T,:T] ==0, float('-inf'))         #(B,T,T)  do trill masking for no-look-ahead. T= Block_size

        #softmax is going to exponentiate the masked_filling (no we know why we tril'ed 0 with -inf)
        wei = F.softmax(wei, dim=-1)                                         #(B,T,T) so soft max on the last dimension T
        #after softmaxing(exponentiation) it becomes
        #[ 1, -inf , -inf] -> [1, 0, 0]
        #[ 1, 0.6 , -inf]  -> [1, 0.6, 0]
        ##[1, 0.6, 0.4] 
        #with '1' being sharper(or stand out more) than 0.6 than 0.4

        wei = self.dropout(wei) #apply dropout mechanism on weights

        #perform the weighted aggregation of the values (add values to the processed key and query)
        v= self.value(x)                #(B, T, hs)
        out = wei @ v                   # (B,T,T) @ (B,T,hs) -> (B,T,hs)      [Matrix Multiply] 
        return out                      



class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention working in parallel (gpu is needed) """
    
    def __init__(self, num_heads,head_size):
        super().__init__()

        #ModuleList() isolates the heads and run them (basically parallelism using gpu) independantly
        #while Sequential() is such that each layer is dependant on the previous one (wait for one to finish before we move onto another)
        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])  #create heads running in parallel using ModuleList (number of heads =4)
        
        # n_embd >= head_size*num_heads, since we have head_size =n_embd//n_head, so to have no dimensionality errors, extend the tensor upto n_embd size
        self.proj = nn.Linear(head_size * num_heads,n_embd)         #projection : project the head_size*num_heads to n_embd ( this adds extra parameters like bias, for the model to learn more)
        
        self.dropout = nn.Dropout(dropout)         #apply dropout mechanism for the linear layer

    def forward(self, x):
         #conncatenate each head together along the last dimesion ie. C 
         #(B,T,C) C (or F- Feature dimesion) is the last dimension = (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3]) (assuming 4 features per head and 3 heads in parallel)
         #features of each head is concatenated to form Channel/Class/Feature dimension for easier processing
        
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out= self.dropout(self.proj(out))          #follow dropout after projection
        return out
    
class FeedForward(nn.Module):
    """a simple linear layer followed by non-linearity. This is done so that the linearity of the input is broken."""

    def __init__(self, n_embd):
        super().__init__()
        #create a neural network
        
        self.net = nn.Sequential(
             nn.Linear(n_embd, 4 * n_embd),          #add Linear layer (input layer with n_embd no. of neurons and hidden layer having 4 times n_embd number of neurons)
             nn.ReLU(),                         # apply rectifying Linear Unit (activation function inside the neuron), ( remember rectifier function: max(0, x) )
             nn.Linear(4* n_embd,n_embd),       # add Linear Layer ( convert 4*n_embd neurons to n_embd neurons of output layer )
             nn.Dropout(dropout),               #a dropout mechanism to drop a part of total neurons for efficient training. dropout = fraction of neurons to drop at random (prevents overfitting)
        )

    def forward(self,x):
        return self.net(x)                  #send input through our neural net and return the outputs from the output layer


class Block(nn.Module):
    """Transformer block(Decoders): Communication followed by computation """
    def __init__(self, n_embd, n_head):

        #n_embd: embedding dimension, n_head = the number of heads we'd like, to learn our data (example: for different POV's and opinions)
        
        super().__init__()
        head_size =n_embd//n_head                        #number of features that each head would be capturing
        self.sa = MultiHeadAttention(n_head,head_size)   #Self attention function done by multi heads is what this means

        self.ffwd = FeedForward(n_embd)                   # -> Linear(Neural network) -> ReLu -> Linear(Neural network) ->

        #For "Residual connection" before and after Feed Forward
        self.ln1 = nn.LayerNorm(n_embd)                 #two layer norms
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        #Note that we are implementing post-norm here, that is "Add and Normalize" and not "Normalize and Add" (which is pre-norm)
        #post-norm is the original architecture proposed for decoder-only-gpt and has better performance for our dataset (try pre-norm too and check)
        
        y = self.sa(x)           #implement self attention
        x= self.ln1(x+y)           #add and norm
        y= self.ffwd(x)             #feed forward
        x= self.ln2(x+y)          #add and norm
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)    #embedding table of the bigrams, n_embd is the vector size if the embedding vector of each token
        

        """positional encoding: for even position we apply the formula: PE(pos,2i) = sin(pos/(10000^(2i/dmodel)))
           And for odd position we apply PE(pos,2i +1) = cos(pos/(10000^(2i/dmodel)))"""

        #self.position_embedding_table = nn.Embedding(block_size,n_embd) 
        self.position_encoding = self._generate_positional_encoding(block_size, n_embd)
        
        #add extra layers for decoders
        self.blocks =nn.Sequential(*[Block(n_embd,n_head=n_head) for i in range(n_layer)]) #create a number of decoder blocks (4)

        self.ln_f = nn.LayerNorm(n_embd)   #final layer norm (layer norm final) for loss convergence 
        self.lm_head = nn.Linear(n_embd,vocab_size)    #langmodel head (the final linear layer for softmax to work with)

        self.apply(self._init_weights)  #apply initial weights (its a formal procedure to be followed, weights defined in nn.Module by experts)

    def _generate_positional_encoding(self, block_size, d_model):
        # Generate sinusoidal positional encoding
        position = torch.arange(0, block_size, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * 
                             -(math.log(10000.0) / d_model))
        pe = torch.zeros(block_size, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        return pe
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight,mean = 0.0, std=0.02)   #std deviation to be set, to make sure weights are taken appropriately
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean =0.0, std=0.02)
        
    def forward(self, index,targets=None):
        # we use logits = self.token_embedding_table(index)   #the normalized probability distributions  for bigrams
        B, T= index.shape
        
        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index)  #(B,T,C)
        #pos_emb = self.position_embedding_table(torch.arange(T,device = device))   #(T,C) remember that arange() fn produces the indices of a tensor
        pos_emb = self.position_encoding[:T,:]
        
        x= tok_emb + pos_emb  #(B,T,C) #following broadcasting semantics of torch, we can add the tables as they are broadcastable
        x= self.blocks(x)  #feed the pos embedded inputs to the 4 layers/bloacks of decoders we constructed earlier
        x =self.ln_f(x)     #feed the result of the decoders to the final linear layer (and to give it to the sofrmax block)

        logits = self.lm_head(x)  #(B,T, vocab_size)

        if targets is None:
            loss= None
        else:
            #Batch dimension, Time dimension (we dont know yet), Channels or Class dimension - vocab size
            B,T,C = logits.shape
            
            logits = logits.view(B*T, C)  #blend B and T (alter shape of the tensor according to the input expectation of cross_entropy() using view() ) 
            targets = targets.view(B*T)       
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        #index is the (B,T) array of indices in the current context

        for i in range(max_new_tokens):
            #get the prediction
            index_cond = index if index.size(1) <= block_size else index[:, -block_size:]
            logits, loss = self.forward(index_cond)

            #focus on last time step alone
            logits = logits[:,-1,:] #becomes (B,C)

            #apply softmax to get probabilities
            probs = F.softmax(logits,dim=-1) 
            #sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #append sampled index to the running sequence
            index=torch.cat((index,index_next),dim=1) #(B,T+1)
        return index
model = GPTLanguageModel(vocab_size)
m = model.to(device)




        




# %%
# optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate, weight_decay=0.01)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
# for iter in range(max_iters):
#     if iter%eval_iters == 0:
#         losses = estimate_loss()
#         print(f"step:{iter},train losses:{losses['train']:.3f} val losses: {losses['val']:.3f}")
#     #sample a batch of data
#     xb, yb = get_batch('train')

#     logits,loss = model.forward(xb,yb)
#     optimizer.zero_grad(set_to_none=True)        #not for RNN's, previously accumulated gradients wont affect current training if set_to_none is True
#     loss.backward()
    
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#     optimizer.step()
#     scheduler.step()

# print(loss.item())

    


# %%
# with open('model-02.pkl','wb') as f:
    # pickle.dump(model,f)

# %%
#context = torch.zeros((1,1), dtype = torch.long, device=device)
#generated_chars = decode(m.generate(context,max_new_tokens=500)[0].tolist())
#print(generated_chars)

# %%
with open('model-02.pkl','rb') as f:
    model=pickle.load(f)
m= model.to(device)
print('loaded sucessfully')

# %%




while True:
    prompt = input("Enter Prompt:\n")
    context = torch.tensor(encode(prompt),dtype=torch.long,device=device)
    generated_chars= decode(m.generate(context.unsqueeze(0),max_new_tokens=150)[0].tolist())    #unsqueeze converts tensor [[1,2,3,4]] to [1,2,3,4]
    print(f"Completion:\n{generated_chars}")