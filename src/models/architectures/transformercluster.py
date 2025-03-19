#rnn style implementation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fast_transformers.transformers
from fast_transformers.attention import AttentionLayer, ClusteredAttention


from fast_transformers.attention import AttentionLayer


MAX_NUM_ACTIONS = 6
NUM_CLUSTERS = 10

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# only for ablation / not used in the final model
class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1/(lengths[..., None]-1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)
    



class Encoder_TRANSFORMERCLUSTER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", **kargs):
        super().__init__()
        
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames #todo
        self.num_classes = num_classes
        
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        
        self.input_feats = self.njoints*self.nfeats
        
        if self.ablation == "average_encoder":
            self.mu_layer = nn.Linear(self.latent_dim, self.latent_dim)
            self.sigma_layer = nn.Linear(self.latent_dim, self.latent_dim)
        else:
            self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
            self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes+1, self.latent_dim))
        
        self.mu_layer1 = nn.Linear(self.latent_dim*MAX_NUM_ACTIONS, self.latent_dim)
        self.sigma_layer1 = nn.Linear(self.latent_dim*MAX_NUM_ACTIONS, self.latent_dim)
        
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        self.seqTransEncoder = fast_transformers.transformers.TransformerEncoder([
            fast_transformers.transformers.TransformerEncoderLayer(
                AttentionLayer(
                    ClusteredAttention(
                        clusters = NUM_CLUSTERS
                    ),
                    self.latent_dim,
                    self.num_heads
                ),
                self.latent_dim,
                self.num_heads
            )
            for i in range(self.num_layers)
        ])
                

    def forward(self, batch):
        x, y, mask  = batch["x"], batch["y"], batch["mask"]
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        #x = x.reshape(bs, njoints*nfeats, nframes)
        # embedding of the skeleton
        x = self.skelEmbedding(x)
        
        self.muQuery[y]

        mu_enc = self.muQuery[y].flatten(1)
        sig_enc = self.sigmaQuery[y].flatten(1)

        mu_enc = self.mu_layer1(mu_enc)
        sig_enc = self.sigma_layer1(sig_enc)
        
        xseq = torch.cat((mu_enc.unsqueeze(0), sig_enc.unsqueeze(0), x), axis=0)

            # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        # create a bigger mask, to allow attend to mu and sigma
        muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)
        maskseq = torch.cat((muandsigmaMask, mask), axis=1)

        final = self.seqTransEncoder(xseq)
        mu = final[0]
        logvar = final[1]
        
        return {"mu": mu, "logvar": logvar}




class Decoder_TRANSFORMERCLUSTER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                 ablation=None, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation

        self.activation = activation
                
        self.input_feats = self.njoints*self.nfeats

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            self.ztimelinear = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)
        else:
            self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        
        self.actionBiasesLayer = nn.Linear(self.latent_dim*MAX_NUM_ACTIONS, self.latent_dim)

        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            self.sequence_pos_encoder = TimeEncoding(self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        # seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
        #                                                   nhead=self.num_heads,
        #                                                   dim_feedforward=self.ff_size,
        #                                                   dropout=self.dropout,
        #                                                   activation=activation)
        # self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
        #                                              num_layers=self.num_layers)

        
    #         """
    # def __init__(self, attention, d_model, n_heads, d_keys=None,
    #              d_values=None, event_dispatcher=""):
    #     super(AttentionLayer, self).__init__()

    #     # Fill d_keys and d_values
    #     d_keys = d_keys or (d_model//n_heads)
    #     d_values = d_values or (d_model//n_heads)

        
        self.seqTransDecoder = fast_transformers.transformers.TransformerDecoder([
            fast_transformers.transformers.TransformerDecoderLayer(
                 AttentionLayer(
                    ClusteredAttention(
                        clusters = NUM_CLUSTERS
                    ),
                    self.latent_dim,
                    self.num_heads
                ),  # self
                 AttentionLayer(
                    ClusteredAttention(
                        clusters = NUM_CLUSTERS
                    ),
                    self.latent_dim,
                    self.num_heads
                ),  # cross
                self.latent_dim
            )
            for i in range(self.num_layers)
        ])
        
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)
        
    
    def forward(self, batch):
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]

        latent_dim = z.shape[1]
        #latent_dim = z.shape[2] #TODO
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            yoh = F.one_hot(y, self.num_classes)
            z = torch.cat((z, yoh), axis=1)
            z = self.ztimelinear(z)
            z = z[None]  # sequence of size 1
        else:
            # only for ablation / not used in the final model
            if self.ablation == "concat_bias":
                # sequence of size 2
                z = torch.stack((z, self.actionBiases[y]), axis=0)
            else:
                z = z + self.actionBiasesLayer(self.actionBiases[y].flatten(1))
                z = z[None]  # sequence of size 1
        
        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        
        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            timequeries = self.sequence_pos_encoder(timequeries, mask, lengths)
        else:
            timequeries = self.sequence_pos_encoder(timequeries)

        #output = self.seqTransDecoder(timequeries, memory=z[0].unsqueeze(1))#,
                                      #tgt_key_padding_mask=~mask)
        output = self.seqTransDecoder(timequeries.permute(1,0,2), memory=z.permute(1,0,2)).permute(1,0,2)
        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)
        
        batch["output"] = output
        return batch
    