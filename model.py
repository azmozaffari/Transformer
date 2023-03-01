import torch 
import torch.nn as nn
import torch.nn.functional as F


        


class MLA(nn.Module):


    def __init__(self, emb_size, num_heads, dropout, bias):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads       

        self.keys = nn.Linear(emb_size, emb_size*num_heads,bias)   # For the output as we have the multi-head that should be concatinated therefore we multiply the output with the number of multi-head
        self.queries = nn.Linear(emb_size, emb_size*num_heads,bias)  
        self.values = nn.Linear(emb_size, emb_size*num_heads,bias)
        
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size*num_heads, emb_size)
        self.lin_drop = nn.Dropout(dropout)

        self.scale = emb_size**(-0.5)

    def forward(self,x):
        
        k = self.keys(x)   #(n_samples, n_patches+1, (embed+1)*head)
        q = self.queries(x)
        v = self.values(x)

        k_ = k.reshape(x.shape[0],x.shape[1],self.emb_size,self.num_heads) #(n_samples, n_patches+1, embed+1, head)
        q_ =  q.reshape(x.shape[0],x.shape[1],self.emb_size,self.num_heads)#(n_samples, n_patches+1, embed+1, head)
        v_ =  v.reshape(x.shape[0],x.shape[1],self.emb_size,self.num_heads)#(n_samples, n_patches+1, embed+1, head)
        
        k = k_.permute(0,3,1,2)#(n_samples, head,n_patches+1, embed+1)
        q = q_.permute(0,3,1,2)#(n_samples, head,n_patches+1, embed+1)
        v = v_.permute(0,3,1,2)#(n_samples, head,n_patches+1, embed+1)
        

        q_ = q.transpose(2,3) #(n_samples, head,  embed+1,n_patches+1,)
        
        
        
        matrix = k@q_  #(n_samples,head, n_patches+1,n_patches+1)
        
        
        matrix = matrix*self.scale
        x = F.softmax(matrix,dim=2)@v  #(n_samples,head, n_patches+1, embed+1)
        x = self.att_drop(x)
        x = x.permute(0,2,1,3)
        x = x.reshape(x.shape[0], x.shape[1],x.shape[2]* x.shape[3]) #(n_samples, patches+1, emb+1)
        x = self.projection(x)
        x = self.lin_drop(x)



        return x



class MLP(nn.Module):
    '''
    Parameters:
        
    Attributes
    '''
    
    
    def __init__(self,n_patches, input_features, hidden_features,output_features, bias, p ):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_features, bias)
        self.Gelu = nn.GELU()
        self.drop = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden_features, output_features, bias)
        self.batch_norm_1 = nn.BatchNorm1d(n_patches)
        self.batch_norm_2 = nn.BatchNorm1d(n_patches)
        

    def forward(self,x):
        x = self.fc1(x)
        
        # print( x.shape)
        x = self.batch_norm_1(x)        
        x = self.Gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.batch_norm_2(x)        
        x = self.drop(x)
        return x
    


class Block(nn.Module):
    #(n_samples, embed+1,n_patches+1)
    def __init__(self, n_patches, n_block, emb_size,num_heads,p, bias,hidden_features  ):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size+1, eps = 1e-6)
        self.mla = MLA(emb_size+1, num_heads, p, bias)
        self.mlp = MLP(n_patches,emb_size+1, hidden_features,emb_size+1, bias, p)
    
    def forward(self,x):

        x = x+self.mla(self.norm(x))
        x = x+self.mlp(self.norm(x))
        return x


class Patch(nn.Module):
    ''' 
    we assumed that the given image is squer shape and the size is divided by 16 which is the patch size
    '''

    def __init__(self, in_channels, emb_size, patch_size): # emb_size is the embedding size without the position
        super().__init__()
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.embedding = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.ones(emb_size,1), requires_grad = False).cuda()#******************************
        

    def forward(self, x):
        s = x.shape[2]//self.patch_size
        x = self.embedding(x)
        position = torch.Tensor(range(0,(s*s)+1)).cuda()
        
        batch = x.shape[0]
        feature = x.shape[1]
        w = x.shape[2]
        h = x.shape[3]

        cls_token = self.cls_token.repeat(x.shape[0],1,1)
        pos_tokens = position.repeat(x.shape[0],1,1)
        x = x.reshape(batch, feature,w*h) # (n_samples, embed_features, n_patches )
        x = torch.cat([cls_token,x], dim = 2)
        x = torch.cat([pos_tokens, x], dim=1) #(n_samples, embed_features+1, n_patches)
        x = x.transpose(1,2)
        return x
    


class VIT(nn.Module):
    def __init__(self, n_patches, n_block, num_heads,p, bias,hidden_features , in_channels, emb_size, patch_size,n_classes):
        super().__init__()
        self.patch = Patch( in_channels, emb_size, patch_size)
        self.blocks = nn.ModuleList([Block(n_patches, n_block, emb_size,num_heads,p, bias,hidden_features) for i in range(n_block)])
        self.fc = nn.Linear(emb_size+1, n_classes)

    def forward(self,x):
        x = self.patch(x) # (n_samples, embed_features+1, n_patches+1)

        for block in self.blocks:
            x = block(x)
        cls_token_final = x[:,0]
        output = self.fc(cls_token_final)
        return output












