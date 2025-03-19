#so far the class does not differ from PROX for a single action
from genericpath import exists
import os
import joblib 
import csv
import pickle
import numpy as np
import torch
from collections import defaultdict
#from Body25_pairs import JOINT_PAIRS_ARR
DIST_THRESHOLD = 2.2
NUM_CONF_THRESHOLD = 13
GOOD_SEQ_THRESHOLD = 0
CONF_THRESH = 0.1
NUM_SEQ_FRAME_THRESH = 10


def main():
    
    datapath="/home/ubuntu/datasets/PROX"
    smpl_processed_path = os.path.join(datapath, 'PROXD')
    action_annot_path = 'annotations'
    dummy_full_path = os.path.join(smpl_processed_path, 'N0Sofa_00034_01/results/s001_frame_01509__00.00.50.268/000.pkl')
    split = 'train'
    all_processed_data = []


    with open(os.path.join(action_annot_path, 'action_annotations_%s.csv')%split, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        sample_id = 0
        for row in reader:
            video_name = row['video_name']
                
            print('video name %s' %video_name)
            actions = row['actions']
            print('current processed video %s' %video_name)
            annot_dir = os.path.join(smpl_processed_path, '%s/results/'%video_name)
           
            annot_files = sorted(os.listdir(annot_dir))
            print('len of annotations files %d' % len(annot_files))
            seq_inds, seq_labels = process_action(actions)
            has_nan = False
            if seq_inds is not None and len(seq_inds) > 0:
                for l in range(len(seq_inds)): 
                    current_subseq = seq_inds[l]
                    parsed_data = defaultdict(list)
                    for frame_i in range(current_subseq[0]-1, current_subseq[1]-1):
                        if frame_i >= len(annot_files):
                            break
                        annot_file = annot_files[frame_i]
                        full_path = os.path.join(annot_dir, '%s/000.pkl'%annot_file)
                        if not os.path.exists(full_path):
                            print("not found %s" %full_path)
                            smpl_data = joblib.load(dummy_full_path) #dummy data
                            smpl_data['body_pose'][...] = float('nan')
                        else:
                            smpl_data = joblib.load(full_path)
                        if np.isnan(smpl_data['body_pose'].sum()):
                            has_nan = True
                            #continue
                        for k in smpl_data.keys():
                            parsed_data[k].append(smpl_data[k])
                    for k in parsed_data.keys():
                        parsed_data[k] = torch.from_numpy(np.asarray(parsed_data[k]))
                        # if k == 'body_pose':
                        #     for m in range(len(parsed_data[k])):
                        #         #print('sum', video_name, m, parsed_data[k][m].sum())
                        #         if np.isnan(parsed_data[k][m].sum()): 
                        #             print(video_name, m, '!!!!nan frame')
                          
                    parsed_data['label'] = seq_labels[l]
                    #print('label', parsed_data['label'])
                    parsed_data['seq_ind'] = current_subseq
                    parsed_data['video_name'] = video_name
                    parsed_data['sample_id'] = sample_id

                    if not has_nan:
                        if len(parsed_data['body_pose']) < 1:
                             continue
                        all_processed_data.append(parsed_data)
                    else:
                        if len(parsed_data['body_pose']) < 1:
                            continue
                        good_subseq_ind = find_good_subseq(parsed_data)
                        seqs_split = generate_good_subseqs(parsed_data, good_subseq_ind)

                        for seq_split in seqs_split:
                            all_processed_data.append(seq_split)
    
    len_sum = 0
    for item1 in all_processed_data:
        len_sum += item1['body_pose'].shape[0]
            
    pickle.dump(all_processed_data, open(os.path.join(datapath, 'processed_data_multi.pkl'), "wb"))


def generate_good_subseqs(parsed_data, good_subseq_ind):
    all_parsed_subseq = []
    for ind in good_subseq_ind:
        if ind[1] - ind[0] < NUM_SEQ_FRAME_THRESH:
            continue
        new_parsed_data = dict()
        for k in parsed_data.keys():
            if k != 'video_name' and k!='sample_id' and k!='seq_ind' and k!='label':
                new_parsed_data[k] = parsed_data[k][ind[0]:ind[1]]
            else:
                new_parsed_data[k] = parsed_data[k]
        new_parsed_data['good_seq_ind'] = ind
        all_parsed_subseq.append(new_parsed_data)
    return all_parsed_subseq


def find_good_subseq(parsed_data):
    seq_poses = parsed_data['body_pose']
    orig_num_frames = len(seq_poses)
    try:
        bad_ind = torch.where(torch.isnan(seq_poses[:,0].sum(dim=1)))[0]
    except:
        print('!!!except')


    bad_ind = torch.sort(bad_ind)[0]

    good_ind = list(set(np.arange(orig_num_frames))-set(bad_ind.numpy()))
   
    if len(good_ind) > 1:
        good_ind= np.asarray(good_ind)
        good_ind = np.sort(good_ind)

        diff_good_array =  [x - good_ind[i - 1] for i, x in enumerate(good_ind)][1:]
        diff_good_array = np.asarray(diff_good_array)
        cut_point_ind = np.where(diff_good_array>1)[0]
        sub_seq_indices = []
        prev_cut_point = -1
        
        for i, c in enumerate(cut_point_ind):
            if i== 0:
                if diff_good_array[c] < 1:
                    sub_seq_indices.append([0 , good_ind[c]])
            else:
                sub_seq_indices.append([good_ind[prev_cut_point+1], good_ind[c]])
            prev_cut_point = c

        if prev_cut_point+1 < len(good_ind):
            sub_seq_indices.append([good_ind[prev_cut_point+1], good_ind[-1]])
        sub_seq_indices = np.asarray(sub_seq_indices)
        #sub_seq_lengths = sub_seq_indices[:,1]-sub_seq_indices[:,0]
        return sub_seq_indices

        
# for the given video annotation (action_data), split it into separate action sequences and their associated label
def process_action(action_data):
    seq_inds = []
    subseq_labels = []
    actions = action_data.split(';')
    for l in range(len(actions)):
        current_action = actions[l]
        #print('current action', current_action)
        #print('current action %s'%current_action)
        split_action = current_action.split(' ')
        subseq_indices = split_action[0].split('-')
        subseq_indices[0] = int(subseq_indices[0].strip())
        subseq_indices[1] = int(subseq_indices[1].strip())
        action_label = int(split_action[1].strip())
        seq_inds.append(subseq_indices)
        subseq_labels.append(action_label)
    return seq_inds, subseq_labels

    

if __name__ == "__main__":
    with torch.no_grad():
        main()


