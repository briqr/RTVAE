import torch
import torch.nn as nn

from ..tools.losses import get_loss_function
from ..rotation2xyz import Rotation2xyz




class CAE(nn.Module):
    def __init__(self, encoder, decoder, device, lambdas, latent_dim, outputxyz,
                 pose_rep, glob, glob_rot, translation, jointstype, vertstrans, disc = None, disc_vert=None,  **kwargs):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        #self.disc = disc
        self.disc_vert = disc_vert

        self.outputxyz = outputxyz
        
        self.lambdas = lambdas
        
        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.device = device
        self.translation = translation
        self.jointstype = jointstype
        self.vertstrans = vertstrans
        
        self.losses = list(self.lambdas) + ["mixed"]

        is_smplx = 'prox' in kwargs['dataset']
        if 'multi' in kwargs['dataset']:
            self.modeltype = 'multi'
        elif 'recurrent' in kwargs['dataset']:
            self.modeltype = 'recurrent'
        elif 'cluster' in kwargs['dataset']:
            self.modeltype = 'cluster'
        else:
            self.modeltype = 'single'
        
        self.rotation2xyz = Rotation2xyz(device=self.device, is_smplx=is_smplx)
        self.param2xyz = {"pose_rep": self.pose_rep,
                          "glob_rot": self.glob_rot,
                          "glob": self.glob,
                          "jointstype": self.jointstype,
                          "translation": self.translation,
                          "vertstrans": self.vertstrans}
        
    def rot2xyz(self, x, mask, **kwargs):
        kargs = self.param2xyz.copy()
        kargs.update(kwargs)
        #print('from CAE x shape', x.shape, mask)
        return self.rotation2xyz(x, mask, **kargs)
    
    def forward(self, batch):
        if self.outputxyz:
            batch["x_xyz"] = self.rot2xyz(batch["x"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]
        # encode
        batch.update(self.encoder(batch))
        # decode
        batch.update(self.decoder(batch))
        # if we want to output xyz
        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        return batch

    def compute_loss(self, batch):
        mixed_loss = 0
        losses = {}
        for ltype, lam in self.lambdas.items():
            #if 'rc' in ltype:
            #    continue
            loss_function = get_loss_function(ltype)
            loss = loss_function(self, batch)
            mixed_loss += loss*lam
            losses[ltype] = loss.item()
        losses["mixed"] = mixed_loss.item()
        return mixed_loss, losses

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def generate_one(self, cls, duration, fact=1, xyz=False):
        y = torch.tensor([cls], dtype=int, device=self.device)[None]
        lengths = torch.tensor([duration], dtype=int, device=self.device)
        mask = self.lengths_to_mask(lengths)
        z = torch.randn(self.latent_dim, device=self.device)[None]
        
        batch = {"z": fact*z, "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)

        if not xyz:
            return batch["output"][0]
        
        output_xyz = self.rot2xyz(batch["output"], batch["mask"])

        return output_xyz[0]
            
    def generate(self, classes, durations, nspa=1,
                 noise_same_action="random", noise_diff_action="random",
                 fact=1, action_timestamps=None, frame_act_map=None):
        if nspa is None:
            nspa = 1
        nats = len(classes)
        
        if self.modeltype == 'single':
            y = classes.to(self.device).repeat(nspa)  # (view(nspa, nats))
        elif self.modeltype == 'multi':
            y = classes.to(self.device).repeat(nspa,1).view(classes.shape[0],-1, classes.shape[1])
        else: #recurrent
            y = classes.to(self.device)#.repeat(nspa,1).view(classes.shape[0],-1, classes.shape[1])


        if self.modeltype == 'single':
            if len(durations.shape) == 1:
                lengths = durations.to(self.device).repeat(nspa)
            else:
                lengths = durations.to(self.device).reshape(y.shape)
        else:
            lengths = durations.to(self.device).flatten()
        
        mask = self.lengths_to_mask(lengths)
        
        if noise_same_action == "random":
            if self.modeltype != 'recurrent':
                if noise_diff_action == "random":
                    z = torch.randn(nspa*nats, self.latent_dim, device=self.device)
                elif noise_diff_action == "same":
                    z_same_action = torch.randn(nspa, self.latent_dim, device=self.device)
                    z = z_same_action.repeat_interleave(nats, axis=0)
                else:
                    raise NotImplementedError("Noise diff action must be random or same.")
            elif self.modeltype == 'cluster':
                z = torch.randn(1, self.latent_dim, device=self.device)
            elif self.modeltype == 'recurrent':
                z = torch.randn(len(classes), self.latent_dim, 1, device=self.device)
            else:
                z = torch.randn(y.shape[1], self.latent_dim, device=self.device)
        elif noise_same_action == "interpolate":
            if noise_diff_action == "random":
                z_diff_action = torch.randn(nats, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(nats, 1)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
            interpolation_factors = torch.linspace(-1, 1, nspa, device=self.device)
            z = torch.einsum("ij,k->kij", z_diff_action, interpolation_factors).view(nspa*nats, -1)
        elif noise_same_action == "same":
            if noise_diff_action == "random":
                z_diff_action = torch.randn(nats, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(nats, 1)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
            z = z_diff_action.repeat((nspa, 1))
        else:
            raise NotImplementedError("Noise same action must be random, same or interpolate.")
        if self.modeltype == 'recurrent':
            #z = torch.randn(1, len(classes[0]), 1, self.latent_dim, device=self.device)
            z = torch.randn(len(classes), len(classes[0]),  self.latent_dim, device=self.device)
            #mask = torch.ones(mask.shape[0], action_timestamps[0,-1].cpu(),  dtype=torch.bool, device = mask.device )
            mask = torch.ones(mask.shape[0], action_timestamps.max().cpu(),  dtype=torch.bool, device = mask.device )
           
        if self.modeltype =='recurrent' and len(z.shape) == 2: #todo for recurrent
            z = z[None]
        batch = {"z": fact*z, "y": y, "mask": mask, "lengths": lengths, "action_timestamps":action_timestamps, "frame_act_map":frame_act_map}
        batch = self.decoder(batch)
        
        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        
        return batch
    
    def return_latent(self, batch, seed=None):
        return self.encoder(batch)["z"]



    def generate_during_train(self, classes):
        with torch.no_grad():
            nats = len(classes)
            nspa = 1
            if self.modeltype == 'single':
                y = classes.to(self.device).repeat(nspa)  # (view(nspa, nats))
            else:
                y = classes.to(self.device).repeat(nspa,1).view(classes.shape[0],-1, classes.shape[1])


            z = torch.randn(nspa*nats, self.latent_dim, device=self.device)
            
            
            batch = {"z": z, "y": y, "mask": None, "lengths": None}
        batch = self.decoder(batch)
        
        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        
        return batch
    
