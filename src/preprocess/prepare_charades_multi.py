#the way to generate a subsequence now is to take the longest valid subsequence and the associated action labels, without a start and end frame stamp for 
#each action

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
MAX_NUM_ACTIONS = 4
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
            conneceted_seqs, connected_labels = connect_overlapping_subseqs(processed_seq, processed_seq_labels)
            if conneceted_seqs is not None and len(processed_seq) > 0:
                for l in range(len(conneceted_seqs)): 
                    current_subseq = conneceted_seqs[l]
                    frame_ids = np.arange(current_subseq[0], current_subseq[1])
                    parsed_data['pose'] = vibe_data['pose'][frame_ids]
                    parsed_data['joints3d'] = vibe_data['joints3d'][frame_ids]
                    parsed_data['op_pose'] = vibe_data['op_pose'][frame_ids]
                    parsed_data['op_distance'] = vibe_data['op_distance'][frame_ids]
                    parsed_data['label'] = connected_labels[l]
                    all_processed_data.append(parsed_data)
            
    pickle.dump(all_processed_data, open(os.path.join(datapath, 'processed_data_multi%d.pkl'%NUM_SEQ_FRAME_THRESH), "wb"))


# for the given video annotation (action_data), split it into separate action sequences and their associated label by finding which
#sequences overlap with a given action label
def process_action(action_annot):
    action_data = action_annot['actions']
    processed_valid_seqs = []
    subseq_labels = []
    if action_data == '':
        return None, None
   
    for l in range(len(action_annot['valid_subseq'])):
        subseq_indices = action_annot['valid_subseq'][l]

        frame_ids = np.arange(subseq_indices[0], subseq_indices[1]) #subseq here does not refer to single action, just a subseq of the whole video and which can subsume multiple actions
        actions = action_data.split(';')
        #find which action labels overlap with a given valid continueous subsequence by iterating over the action annotation of the whole video
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


def connect_overlapping_subseqs(subseqs, subseq_labels):
    subseqs.sort(key=lambda x: x[0], reverse=False)
    processed_seqs = []
    processed_labels = []
    for ind, seq in enumerate(subseqs):
        sf1 = seq[0]
        ef1 = seq[1]
        new_seq = [sf1,ef1]
        current_labels = []
        current_labels.append(subseq_labels[ind])
        processed_seqs.append(new_seq)
        if ind < (len(subseqs)-1):
            seq2 = subseqs[ind+1]
            sf2 = seq2[0]
            ef2 = seq2[1]
            
            if ef1 > sf2: #these actions overlap
                current_labels.append(subseq_labels[ind+1])
                if ef2 > ef1:
                    new_seq[1] = ef2
            processed_labels.append(np.asarray(current_labels))
        # if ind > 0: #check if there is overlap with the previous action (the overlapping subseq is already appended)
        #     prev_seq = subseqs[ind-1]
        #     prev_ef = prev_seq[1]
        #     if prev_ef > sf1:
        #         new_seq['act_start_time'] = prev_ef
                
    return processed_seqs, processed_labels




if __name__ == "__main__":
    main()


