import numpy as np
import os
import joblib
import json
import torch
import glob
import subprocess
import cv2
import colorsys
import matplotlib
import math
import matplotlib.pyplot as plt
import random
import csv
import pickle
#from Body25_pairs import JOINT_PAIRS_ARR
DIST_THRESHOLD = 2.2
NUM_CONF_THRESHOLD = 13
GOOD_SEQ_THRESHOLD = 0
CONF_THRESH = 0.1
NUM_SEQ_FRAME_THRESH = 60
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

os.environ['PYOPENGL_PLATFORM'] = 'egl'


colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
cmap = matplotlib.cm.get_cmap('hsv')




def main():
    
    smpl_processed_path = '/home/ubuntu/datasets/charades_ego/smpl_processed'
    datapath="/home/ubuntu/datasets/charades_ego"
    action_annot_path = os.path.join(datapath, 'annotations')
    split = 'train'
    all_data = dict()
    all_processed_data = []

    with open(os.path.join(action_annot_path, 'CharadesEgo_v1_%s.csv')%split, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        for row in reader:
            current_data = dict()
            if 'EGO' in row['id']: 
                continue
            video_name = row['id']
            print('current processed video %s' %video_name)
            pkl_name = os.path.join(smpl_processed_path, '%s.pkl'%video_name)
            if not os.path.exists(pkl_name):
                print('video name %s still does not have a vibe and openpose prediction,skipping...' %row['id'])
                continue
            vibe_data = joblib.load(pkl_name)
            max_len = -1
            for data_key in vibe_data.keys():
                if 'op_pose' in vibe_data[data_key] and len(vibe_data[data_key]) > max_len:
                    max_key = data_key
                    max_len = len(vibe_data[data_key])
            subseq_inds = vibe_data[max_key]['subsequence_indices']
            vibe_data = vibe_data[max_key]
            sub_seq_lengths = subseq_inds[:,1]- subseq_inds[:,0]
            valid_subseq_ind = subseq_inds[sub_seq_lengths>NUM_SEQ_FRAME_THRESH]
            
            if len(valid_subseq_ind) < 1:
                continue

            for field in fieldnames:
                #if field == 'id':
                #    #id = current_data
                #    continue
                if field != 'id':
                    current_data[field] = row[field]
            
            current_data['valid_subseq'] = valid_subseq_ind
            parsed_data = dict()
            processed_seq, processed_seq_labels = process_action(current_data)
            if processed_seq is not None and len(processed_seq) > 0:
                for l in range(len(processed_seq)): 
                    current_subseq = processed_seq[l]
                    frame_ids = np.arange(current_subseq[0], current_subseq[1])
                    parsed_data['pose'] = vibe_data['pose'][frame_ids]
                    parsed_data['joints3d'] = vibe_data['joints3d'][frame_ids] 
                    parsed_data['op_pose'] = vibe_data['op_pose'][frame_ids]
                    parsed_data['op_distance'] = vibe_data['op_distance'][frame_ids]
                    parsed_data['label'] = processed_seq_labels[l]
                    all_processed_data.append(parsed_data)
            
    pickle.dump(all_processed_data, open(os.path.join(datapath, 'processed_data_%d.pkl'%NUM_SEQ_FRAME_THRESH), "wb"))



def process_action(action_annot):
    action_data = action_annot['actions']
    processed_valid_seqs = []
    subseq_labels = []
    if action_data == '':
        return None, None
   
    for l in range(len(action_annot['valid_subseq'])):
        subseq_indices = action_annot['valid_subseq'][l]

        frame_ids = np.arange(subseq_indices[0], subseq_indices[1])   
        actions = action_data.split(';')

        max_act_len = -1
        for action in actions:
            if action == '':
                continue        
            lower_same_act = subseq_indices[0]
            higher_same_act = subseq_indices[1]
            action_split = action.split(' ')
            act_label = action_split[0]
            act_start_frame = int(float(action_split[1]) * 30)
            act_end_frame = int(float(action_split[2]) * 30)
            if (min(frame_ids) >= act_start_frame and min(frame_ids) < act_end_frame) or (act_start_frame >=min(frame_ids) and act_end_frame <=max(frame_ids)):
                if act_end_frame - act_start_frame > max_act_len:
                    if act_start_frame > lower_same_act:
                        lower_same_act = act_start_frame
                    if act_end_frame < higher_same_act:
                        higher_same_act = act_end_frame
                    max_act_len = act_end_frame - act_start_frame
                    
                temp_frame_ids = frame_ids[(frame_ids >= lower_same_act) & (frame_ids <=higher_same_act) ]
                if len(temp_frame_ids) > NUM_SEQ_FRAME_THRESH:
                    updated_seq_indices = [min(temp_frame_ids), max(temp_frame_ids)]
                    processed_valid_seqs.append(updated_seq_indices)
                    subseq_labels.append(act_label)
    
    return processed_valid_seqs, subseq_labels

        

def visualise_smpl_op_aligned_2d(vibe_data, orig_vid_name, op_poses, all_good_ind, save_path, flip=False):
    cap = cv2.VideoCapture(orig_vid_name)
    out = None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    renderer = None
    mesh_color_red = colorsys.hsv_to_rgb(1., 0.5, 1.0)
    mesh_color_yellow = colorsys.hsv_to_rgb(1./3, 0.5, 1.0) #np.random.rand(), 0.5, 1.0)
    for fno in range(0, total_frames, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        _, oriImg = cap.read()
        if out is None:
            codec ='MJPG'
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(save_path, 0x7634706d, 30, ([oriImg.shape[1], oriImg.shape[0]]))

        if True and renderer is None:
            img_shape = oriImg.shape[:2]
            orig_height, orig_width = img_shape[:2]
            renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=False)
        if fno > len(op_poses)-1: # or fno > len(vibe_data['verts'])-1:
            break
        if not fno in all_good_ind:
            meshc = mesh_color_red
        else:
            meshc = mesh_color_yellow
        if fno in vibe_data['frame_ids']:
            frame_verts = vibe_data['verts'][fno]
            frame_cam = vibe_data['orig_cam'][fno]
            mesh_filename = None

            oriImg = renderer.render(
                oriImg,
                frame_verts,
                cam=frame_cam,
                color=meshc,
                mesh_filename=mesh_filename,
            )

        
        smpl_2d_pose = vibe_data['joints2d_img_coord'][fno] #now that joints2d is filled with dummy data #[op_s+smpl_good_start]
        poses = op_poses[fno]
        canvas = oriImg.copy()
       
        col =  colors[7]
        col1 = colors[15]
        for j in range(min(poses.shape[0]-1, 27)):
            rgba = np.array(cmap(1 - j/18. - 1./36))
            rgba[0:3] *= 255
            if poses[j].all == 0:
                continue
            if np.isnan(poses[j]).any() or poses[j, 0] > oriImg.shape[1] or poses[j][0]>oriImg.shape[1]:
                continue
            x, y = poses[j][0:2]
            x1, y1 = smpl_2d_pose[j]
            if flip:
                cv2.circle(canvas, (int(y),int(x)), 4, col, thickness=-1)
                cv2.circle(canvas, (int(y1),int(x1)), 2, col1, thickness=-1)
            else:
                cv2.circle(canvas, (int(x), int(y)), 4, col, thickness=-1)
                cv2.circle(canvas, (int(x1), int(y1)), 2, col1, thickness=-1)

       
        out.write(canvas)
    out.release()
    a = 0






if __name__ == "__main__":
    main()


