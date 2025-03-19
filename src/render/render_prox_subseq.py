import numpy as np
import imageio
import os
import argparse
from tqdm import tqdm
from src.render.renderer import get_renderer
import joblib
from src.models.smpl import SMPLX
import torch
import imageio
import pickle
import cv2
import imageio
from src.datasets.prox_recurrent import init_class_map



def main():
    torch.cuda.set_device(1)
    device = torch.device("cuda")
    smpl_model = SMPLX(batch_size=1).eval().to(device)#TODO SMPL

    width = 1024
    height = 1024
    background = np.zeros((height, width, 3))
    renderer = get_renderer(width, height, faces=smpl_model.body_model.faces)
    datapath="/home/ubuntu/datasets/PROX"
    data = pickle.load(open(os.path.join(datapath, 'processed_data_train.pkl'), 'rb'))
    vns = os.listdir(os.path.join(datapath, 'PROXD'))
    cam=(0.75, 0.75, 0, 0.10)
    color=[0.11, 0.53, 0.8]
    class_file = os.path.join(datapath, 'annotations/action_labels.txt')

    prox_action_enumerator = dict()
    class_map = init_class_map()

    with open (class_file, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            label_actname = line.split(':')
            label = int(label_actname[0])

            action = label_actname[1]
            label = class_map[label]
            prox_action_enumerator[label] = action


    for data_i, vid_data in enumerate(data):
        vn = vid_data['video_name']
        savepath = 'vis_prox'

        if not os.path.exists(savepath):
            os.mkdir(savepath)

        if len(vid_data['body_pose']) < 1:
            continue
        rotations = vid_data['body_pose'].to(device)
        global_orient =vid_data['global_orient'].to(device)
        smpl_model = SMPLX(batch_size=len(rotations)).eval().to(device)#TODO SMPL
        render_res = smpl_model(return_verts=True, body_pose=rotations, global_orient=global_orient)


        all_verts =  render_res['vertices'].detach().cpu().numpy()
        #save_vid_name = os.path.join(savepath, '%s_label%s.mp4'%(vn,vid_data['label']))
        #out = cv2.VideoWriter(save_vid_name, 0x7634706d, 30, ([width, height]))
        for l in range(len(all_verts)):
            image = renderer.render(background, all_verts[l], cam, color=color)
            save_im_name = os.path.join(savepath, '%s_%s_%d.png'%(vn,prox_action_enumerator[vid_data['label']][0:40],l))
            imageio.imwrite(save_im_name, image)
            #out.write(image)
        #out.release()



if __name__ == "__main__":
    main()
