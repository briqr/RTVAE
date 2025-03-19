#rnn style implementation
import numpy as np
from sklearn.metrics import jaccard_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_transformers.builders import RecurrentEncoderBuilder, RecurrentDecoderBuilder


attention_type_enc = 'linear' 
attention_type_dec_cross = 'linear' 
attention_type_dec_self = 'linear' 
fac = 1.0
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

    def forward(self, x, last_pos=0, offset =2):
        # not used in the final model
        if last_pos == 0:
            start_i = last_pos
        else:
            #start_i = last_pos + last_pos*offset
            start_i = last_pos + offset
        x = x + self.pe[start_i:start_i+x.shape[0], :]
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
    



class Encoder_TRANSFORMERRECURRENT(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", **kargs):
        super().__init__()
        print('**attention types**', attention_type_enc, attention_type_dec_cross, attention_type_dec_self)

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
            self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        added_future = 0
        self.seqTransEncoder = RecurrentEncoderBuilder.from_kwargs(
        attention_type = attention_type_enc,
        n_layers=num_layers,
        n_heads=self.num_heads,
        feed_forward_dimensions=ff_size,
        query_dimensions=self.latent_dim//num_heads*(3+added_future), # todo, plus 2 for the next mu sigma query
        value_dimensions=self.latent_dim//num_heads*(3+added_future)#  todo
    ).get()

                

    def forward(self, batch):
        x, y, mask, act_tstamps, frame_act_map  = batch["x"], batch["y"], batch["mask"], batch['action_timestamps'], batch['frame_act_map']
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        #x = x.reshape(bs, njoints*nfeats, nframes)\
        # embedding of the skeleton
        x = self.skelEmbedding(x)

        # only for ablation / not used in the final model
        if self.ablation == "average_encoder":
            # add positional encoding
            x = self.sequence_pos_encoder(x)
            
            # transformer layers
            final = self.seqTransEncoder(x, src_key_padding_mask=~mask)
            # get the average of the output
            z = final.mean(axis=0)
            
            # extract mu and logvar
            mu = self.mu_layer(z)
            logvar = self.sigma_layer(z)
        else:
            # adding the mu and sigma queries
            mu_query = self.muQuery[y]
            sig_query = self.sigmaQuery[y]
            

            # add positional encoding
            #xseq = self.sequence_pos_encoder(xseq)

            # create a bigger mask, to allow attend to mu and sigma
            muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)
            maskseq = torch.cat((muandsigmaMask, mask), axis=1)
            state = None
            mu = []
            logvar = []

            max_t = torch.zeros(bs).to(x.device)
            for t in range(0, act_tstamps.max()):
                ind = torch.zeros(bs).to(x.device).long()
                for s,f_map in enumerate(frame_act_map):
                    if t in f_map:
                        ind[s] = f_map[t]
                        max_t[s] = f_map[t]
                    else:
                        ind[s] = max_t[s]
                
                current_mu_query = torch.cat([torch.index_select(a,0, i).unsqueeze(0) for a, i in zip(mu_query, ind)])
                current_sig_query = torch.cat([torch.index_select(a,0, i).unsqueeze(0) for a, i in zip(sig_query, ind)])
                xseq = torch.cat((current_mu_query.permute(1,0,2), current_sig_query.permute(1,0,2), x[t:t+1,:]), axis=0)

                xseq = self.sequence_pos_encoder(xseq, last_pos=t, offset=2) # plus 2 for mu and sigma
                xseq = xseq.permute(1,0,2).flatten(1)
                final,state = self.seqTransEncoder(xseq, state=state) #src_key_padding_mask=~maskseq)
                if t == 0:
                    state = np.asarray(state)
                mu.append(final[:,0:self.latent_dim])
                logvar.append(final[:, self.latent_dim:self.latent_dim+self.latent_dim])
            mu = torch.stack(mu)
            logvar = torch.stack(logvar)
            mus = []
            logvars = []
            act_tstamps_b = (act_tstamps-1).permute(1,0)
            act_tstamps_b[act_tstamps_b<0] = 0

            for s,t_s in enumerate(act_tstamps_b):
                if True: # not the averaged stat, just the last one
                    current_m = torch.cat([torch.index_select(a,0, i).unsqueeze(0) for a, i in zip(mu.permute(1,0,2), t_s)])[:,0]
                    mus.append(current_m)
                    current_l = torch.cat([torch.index_select(a,0, i).unsqueeze(0) for a, i in zip(logvar.permute(1,0,2), t_s)])[:,0]
                    logvars.append(current_l)
                elif False: # use the averaged stats
                    avg_mu = []
                    avg_sig = []
                    for b_i, t in enumerate(t_s):
                        current_m = torch.mean(mu[:t, b_i], dim=0) # for each element from the list of timestamps, there are b timestamps
                        current_m[current_m!=current_m] = 0
                        #print('*************************current_m', current_m)
                        #current_m = torch.cat([torch.index_select(a,0, i).unsqueeze(0) for a, i in zip(mu.permute(1,0,2), t_s)])[:,0]
                        avg_mu.append(current_m)
                        current_l = torch.mean(logvar[:t, b_i], dim=0)
                        current_l[current_l!=current_l] = 0

                        #current_l = torch.cat([torch.index_select(a,0, i).unsqueeze(0) for a, i in zip(logvar.permute(1,0,2), t_s)])[:,0]
                        avg_sig.append(current_l)
                    mus.append(torch.stack(avg_mu))
                    logvars.append(torch.stack(avg_sig))
                

        
        mu = torch.stack(mus).permute(1,0,2)
        logvar = torch.stack(logvars).permute(1,0,2)
        return {"mu": mu, "logvar": logvar}




class Decoder_TRANSFORMERRECURRENT(nn.Module):
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


        self.seqTransDecoder = RecurrentDecoderBuilder.from_kwargs(
        cross_attention_type = attention_type_dec_cross, 
        self_attention_type = attention_type_dec_self,
        n_layers=num_layers,
        n_heads=self.num_heads,
        feed_forward_dimensions=ff_size,
        query_dimensions=self.latent_dim//num_heads,
        value_dimensions=self.latent_dim//num_heads
    ).get()


        
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)
        
    def forward(self, batch):
        z, y, mask, lengths, action_timestamps, frame_act_map = batch["z"], batch["y"], batch["mask"], batch["lengths"], batch["action_timestamps"],batch['frame_act_map']

        latent_dim = z.shape[2] 
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
               
                z = z + self.actionBiases[y] 

                a = 0
        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        if False:
            z = torch.randn(z.shape[0], nframes, self.latent_dim, device=z.device) # only during inference for causal per acton latent vector experiment
            max_t = torch.zeros(bs).to(timequeries.device)
            for t in range(len(timequeries)): #todo, restart the queries to signal a new action
                ind = torch.zeros(bs).to(timequeries.device).long()
                for s,f_map in enumerate(frame_act_map):
                    if t in f_map:
                        ind[s] = f_map[t]
                        max_t[s] = f_map[t]
                    else:
                        ind[s] = max_t[s]
                for b_i in range(bs):
                    current_z_bias = self.actionBiases[y[b_i,ind[b_i]]] #z.shape 1* 5 *256 d
                    z[b_i,t] += current_z_bias 

        
        
        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            timequeries = self.sequence_pos_encoder(timequeries, mask, lengths)
        else:
            timequeries = self.sequence_pos_encoder(timequeries)

        if True:
            state = None
            output = []
            max_t = torch.zeros(bs).to(timequeries.device)
            for t in range(len(timequeries)): #todo, restart the queries to signal a new action
                ind = torch.zeros(bs).to(timequeries.device).long()
                next_ind = torch.zeros(bs).to(timequeries.device).long()
                if False: # use for w/o look-ahead-back strategy
                     for s,f_map in enumerate(frame_act_map):
                        if t in f_map:
                            ind[s] = f_map[t]
                            max_t[s] = f_map[t]
                        else:
                            ind[s] = max_t[s]
                        if (t+1) in f_map:
                            next_ind[s] = f_map[t+1]
                        else:
                            next_ind[s] = ind[s]

                        current_z = torch.cat([torch.index_select(a,0, i).unsqueeze(0) for a, i in zip(z, ind)])


              
                out, state = self.seqTransDecoder(timequeries[t], z, state=state) #here current_z has sequence length 1
                output.append(out)
            output = torch.stack(output)


        
        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        
        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)
        
        batch["output"] = output
        return batch


