import torch
import src.utils.rotation_conversions as geometry

from .smpl import SMPL,SMPLX, JOINTSTYPE_ROOT
from .get_model import JOINTSTYPES
import os
import numpy as np
import cv2
from src.render.renderer import get_renderer

class Rotation2xyz:
    def __init__(self, device, is_smplx=False):
        self.device = device
        self.issmplx = is_smplx

        width = 1024
        height = 1024
        if self.issmplx:
            self.vis_path = 'vis_test_prox'
            if not os.path.exists(self.vis_path):
                os.mkdir(self.vis_path)
            self.smpl_model = SMPLX().eval().to(device)#TODO SMPL
            self.renderer = get_renderer(width, height, faces=self.smpl_model.body_model.faces)

        else:
            self.smpl_model = SMPL().eval().to(device)#TODO SMPL
            self.vis_path = 'vis_train_charades'
            self.renderer = get_renderer(width, height, faces=None)
        

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, **kwargs):
        if pose_rep == "xyz":
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")

        if True and translation:
            x_translations = x[:, -1, :3]
            x_rotations = x[:, :-1]
        else:
            x_rotations = x

        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rotmat":
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
            if self.issmplx: # the smplx model expects the body pose to be represented in axis-angle whereas smpl in rotation matrix.
                rotations = geometry.matrix_to_axis_angle(rotations) 
        else:
            raise NotImplementedError("No geometry for this one.")

        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0:1]
            rotations = rotations[:, 1:]
        if not self.issmplx:
            if betas is None:
                betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                                    dtype=rotations.dtype, device=rotations.device)
                betas[:, 1] = beta
            #import ipdb; ipdb.set_trace()
            #handpose = torch.zeros(rotations.shape[0],2,rotations.shape[2],rotations.shape[3]).to(rotations.device)
            #rotations = torch.cat((rotations,handpose), dim=1)
            out = self.smpl_model(return_verts=True, body_pose=rotations, global_orient=global_orient, betas=betas)
        else:
            self.smpl_model = SMPLX(batch_size=rotations.shape[0]).eval().to(x.device)#TODO SMPL
           
            out = self.smpl_model(return_verts=True, body_pose=rotations[:,:23], global_orient=global_orient) #, betas=betas)
        vis = False
        if vis: # and self.issmplx:
            width = 1024
            height = 1024
            cam=(0.75, 0.75, 0, 0.10)
            color=[0.11, 0.53, 0.8]
            background = np.zeros((height, width, 3))

           
            all_verts =  out['vertices'].detach().cpu().numpy()
            fn = np.random.randint(1900000)
            print('file name %s' %fn)
            savepath = os.path.join('%s/%s.mp4'%(self.vis_path, fn))
            vid_out = cv2.VideoWriter(savepath, 0x7634706d, 30, ([width, height]))

            for s in range(30) :#len(all_verts)): #//60):
                #savepath = os.path.join('%s/%s_%d.mp4'%(self.vis_path, fn,s))
                #vid_out = cv2.VideoWriter(savepath, 0x7634706d, 30, ([width, height]))
                #for l in range(all_verts.shape[1]):
                    #image = self.renderer.render(background, all_verts[60*s+l], cam, color=color)
                image = self.renderer.render(background, all_verts[s], cam, color=color)
                vid_out.write(image)
                #vid_out.release()
            vid_out.release()
        # get the desirable joints
        joints = out[jointstype] 

        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        # the first translation root at the origin on the prediction
        if jointstype != "vertices":
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

        if translation and vertstrans:
            # the first translation root at the origin
            x_translations = x_translations - x_translations[:, :, [0]]

            # add the translation to all the joints
            x_xyz = x_xyz + x_translations[:, None, :, :]

        return x_xyz
