import torch.nn as nn
from .base import BaseVisModel

'''
1. layer norm or batch norm
2. relu or leakyrelu
3. whether to add relu in the last layer?
'''

class BN_AE(BaseVisModel):

    def __init__(self, encoder_dims, decoder_dims):
        super(BN_AE, self).__init__()
  
        assert len(encoder_dims) > 1
        assert len(decoder_dims) > 1
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        # Build Encoder
        modules = list()
        for i in range(0, len(self.encoder_dims)-2):
            modules.append(
                nn.Sequential(
                nn.Linear(self.encoder_dims[i], self.encoder_dims[i+1]),
                nn.BatchNorm1d(self.encoder_dims[i+1]),
                nn.ReLU(True) 
                )
            )
        modules.append(nn.Linear(self.encoder_dims[-2], self.encoder_dims[-1]))
        self.encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = list()
        for i in range(0, len(self.decoder_dims)-2):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.decoder_dims[i], self.decoder_dims[i+1]),
                    nn.BatchNorm1d(self.decoder_dims[i+1]),
                    nn.ReLU(True)
                )
                
            )
        modules.append(nn.Linear(self.decoder_dims[-2], self.decoder_dims[-1]))
        self.decoder = nn.Sequential(*modules)

    def forward(self, edge_to, edge_from):
        outputs = dict()
        embedding_to = self.encoder(edge_to)
        embedding_from = self.encoder(edge_from)
        recon_to = self.decoder(embedding_to)
        recon_from = self.decoder(embedding_from)
        
        outputs["umap"] = (embedding_to, embedding_from)
        outputs["recon"] = (recon_to, recon_from)

        return outputs


class LN_AE(BaseVisModel):

    def __init__(self, encoder_dims, decoder_dims):
        super(LN_AE, self).__init__()
  
        assert len(encoder_dims) > 1
        assert len(decoder_dims) > 1
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        # Build Encoder
        modules = list()
        for i in range(0, len(self.encoder_dims)-2):
            modules.append(
                nn.Sequential(
                nn.Linear(self.encoder_dims[i], self.encoder_dims[i+1]),
                nn.LayerNorm(self.encoder_dims[i+1]),
                nn.ReLU(True) 
                )
            )
        modules.append(nn.Linear(self.encoder_dims[-2], self.encoder_dims[-1]))
        self.encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = list()
        for i in range(0, len(self.decoder_dims)-2):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.decoder_dims[i], self.decoder_dims[i+1]),
                    nn.LayerNorm(self.decoder_dims[i+1]),
                    nn.ReLU(True)
                )
                
            )
        modules.append(nn.Linear(self.decoder_dims[-2], self.decoder_dims[-1]))
        self.decoder = nn.Sequential(*modules)

    def forward(self, edge_to, edge_from):
        outputs = dict()
        embedding_to = self.encoder(edge_to)
        embedding_from = self.encoder(edge_from)
        recon_to = self.decoder(embedding_to)
        recon_from = self.decoder(embedding_from)
        
        outputs["umap"] = (embedding_to, embedding_from)
        outputs["recon"] = (recon_to, recon_from)

        return outputs

class IN_AE(BaseVisModel):

    def __init__(self, encoder_dims, decoder_dims):
        super(IN_AE, self).__init__()
  
        assert len(encoder_dims) > 1
        assert len(decoder_dims) > 1
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        # Build Encoder
        modules = list()
        for i in range(0, len(self.encoder_dims)-2):
            modules.append(
                nn.Sequential(
                nn.Linear(self.encoder_dims[i], self.encoder_dims[i+1]),
                nn.InstanceNorm1d(self.encoder_dims[i+1]),
                nn.ReLU(True) 
                )
            )
        modules.append(nn.Linear(self.encoder_dims[-2], self.encoder_dims[-1]))
        self.encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = list()
        for i in range(0, len(self.decoder_dims)-2):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.decoder_dims[i], self.decoder_dims[i+1]),
                    nn.InstanceNorm1d(self.decoder_dims[i+1]),
                    nn.ReLU(True)
                )
                
            )
        modules.append(nn.Linear(self.decoder_dims[-2], self.decoder_dims[-1]))
        self.decoder = nn.Sequential(*modules)

    def forward(self, edge_to, edge_from):
        outputs = dict()
        embedding_to = self.encoder(edge_to)
        embedding_from = self.encoder(edge_from)
        recon_to = self.decoder(embedding_to)
        recon_from = self.decoder(embedding_from)
        
        outputs["umap"] = (embedding_to, embedding_from)
        outputs["recon"] = (recon_to, recon_from)

        return outputs
    
