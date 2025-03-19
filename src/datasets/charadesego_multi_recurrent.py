import pickle as pkl
import numpy as np
import os
from .dataset import Dataset
import csv
import joblib
import random

import torch
import pickle

from .tools import parse_info_name
from ..utils.tensors import collate
from ..utils.misc import to_torch
import src.utils.rotation_conversions as geometry
from collections import defaultdict
from rangedict import RangeDict


MAX_NUM_CLASSES = 8
MAX_ACT_LEN = 80
class CharadesRecurrent(Dataset):
    dataname = "CharadesEgomultirecurrent"
    
    def __init__(self, datapath="/home/ubuntu/datasets/charades_ego", split='train', **kargs):
        print('*****dataset name charadesego****')
        self.datapath = datapath
        #CharadesEgo.num_not_found = 0
        super().__init__(**kargs)
        class_file = os.path.join(datapath, 'annotations/Charades_v1_classes.txt')
        charadesego_action_enumerator = dict()
        labels = []
        self.max_num_classes = MAX_NUM_CLASSES
       
        action_indices = np.arange(70, 100)
   

        total_num_actions = len(action_indices) #todo
        print('****total num actions %d'% total_num_actions)
        action_indices = action_indices[:total_num_actions]

        self.total_len = 0

        self.max_len = MAX_ACT_LEN
        self.max_num_classes = MAX_NUM_CLASSES

        if 'MAX_ACT_LEN' in kargs:
            self.max_len = kargs['MAX_ACT_LEN']
        if 'MAX_NUM_CLASSES' in kargs:
            self.max_num_classes = kargs['MAX_NUM_CLASSES']
    
        #action_indices = np.arange(12)
        with open (class_file, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if not idx in action_indices:
                    continue
                label = line[0:4]
                action = line[5:]
                charadesego_action_enumerator[label] = action
                labels.append(label)
        #print('avail labels', labels)
        self._action_to_label = {x: i for i, x in enumerate(labels)}
        self._label_to_action = {i: x for i, x in enumerate(labels)}
        self._idx_to_action = {i: charadesego_action_enumerator[x] for i, x in enumerate(labels)}
        #print('idx to action', self._idx_to_action )
        #for i, x in enumerate(labels):
        #    print('label to action', charadesego_action_enumerator[x], self._label_to_action[i])

        #print('charadesego_action_enumerator', charadesego_action_enumerator)


        #print('action to label', self._idx_to_action)
        self.num_attempts = 0
        num_ego_videos = 0
        #self.vibe_pred_path = os.path.join(datapath, 'smpl_pred') #path for vibe and openpose predictions and indices of good sequences
        self.vibe_pred_path = os.path.join(datapath, 'smpl_processed') #path for vibe and openpose predictions and indices of good sequences
        self.annot_path = os.path.join(datapath, 'annotations')
      
        self.data_path = datapath



        self.num_classes = total_num_actions
        #data = pickle.load(open(os.path.join(datapath, 'processed_data30.pkl'), 'rb'))
        data = pickle.load(open(os.path.join(datapath, 'processed_data_30.pkl'), 'rb'))
        #data = data[:6000]
        if split == 'train':
            #data = data[:200]
            data = data[:3500]
        else:
            data = data[3500:3540] + data[3602:3650] + data[4131:4120] + data[5010:5040]
       

        
        #data = np.load('%s.npy'%os.path.join(datapath, 'processed_data'), allow_pickle=True)
        #self.data = data
        #self._test = [0]
        label_num_samples = defaultdict(lambda:0)

        self.data = []
        for data_i, item in enumerate(data):
            #if data_i in self._test:
            #    continue
            if item['label'] in charadesego_action_enumerator:
                self.data.append(item)
                label_num_samples[item['label']] += 1
                #if len(self.data) == 100:
                #    break
        self.samples_weights = []
        for data_i, item in enumerate(self.data):
            weight_i = 1.0/label_num_samples[item['label']]
            self.samples_weights.append(weight_i)
        if False:
            self._test = np.random.randint(0, len(self.data)-1, size=100)
            np.save('charades_test', self._test)
            val_ind = np.random.randint(0,len(self._test)-1, size=60)
            self._val = self._test[val_ind]
            np.save('charades_val', self._val)
        else:
            self._test = np.load('charades_test.npy')
            self._val = np.load('charades_val.npy')
            #todo
            self._test = self._val
            
        all = list(range(len(self.data)))
        self._train = list(set(all) - set(self._test))
        unique_labels =  defaultdict(lambda:0)
        for k in range(len(self.data)):
            unique_labels[self.data[k]['label']] = unique_labels[self.data[k]['label']] + 1
        #print('unique labels counts', unique_labels)




        print('num of samples %d' %self.__len__())
        #self._actions = np.random.randint(0, total_num_actions-1, self.__len__())


        self._action_classes = charadesego_action_enumerator
        # for data_ind in range(len(self.data)):

        #     if (data_ind == len(self.data)-1 or self.data[data_ind]['video_name'] != self.data[data_ind+1]['video_name']):
        #         num_sequences += 1
        #         num_classes_per_seq
        # print('num sequences %d' %num_sequences)

    

      
    def __getitem__(self, index):
        # the different to prox_multi is that here we take all subsequences belonging to a video
        seq_data = []
        labels = []
        if index >= len(self._train):
            index = index % len(self._train)
        data_ind = self._train[index]
        if self.split == 'train':
            num_frames = 0
            while True: 
                current_data = self.data[data_ind]
                seq_data.append(current_data)
                labels.append(current_data['label'])
                num_frames += len(current_data['pose'])
                if False: #todo, experiment recursive using a single action
                    break
                if True and len(labels) == self.max_num_classes:
                    break
                if False and num_frames >= self.max_len and len(seq_data) >= self.max_num_classes:
                    break

                if True and (data_ind == len(self.data)-1 or self.data[data_ind]['video_name'] != self.data[data_ind+1]['video_name']):  #todo disable for testing
                    break
                data_ind = (data_ind + 1)%len(self.data)

        else:
            data_ind = self._test[index]
                    
        if True:
            inp, target, action_tstamps, frame_act_map = self._get_item_data_index(seq_data)
        
        return inp, target, action_tstamps, frame_act_map


    def _get_item_data_index(self, annots):     
        #print('len pose', len(annot['pose']))
        inp = None
        #inp = []
        act_tstamps = []
        targets = []
        total_act_len = 0
        frame_act_map = RangeDict()
        for annot in annots:
            current_act_len = len(annot['pose'])

            if total_act_len + current_act_len >=self.max_len:
                until_max = self.max_len - total_act_len
            else:
                until_max = current_act_len
            if until_max < 1:
                break
            total_act_len += until_max
            current_frame_ix = np.arange(0, until_max) 
            
            current_inp = self.get_pose_data(annot, current_frame_ix) #frame_ix) #
            current_target = self.action_to_label(annot['label'])

            #if len(inp) == 0:
            #print('inp.sum: ', current_inp.sum())
            if inp is None:
                #inp.append(current_inp)
                inp = current_inp
                act_tstamps.append(until_max)
                frame_act_map[(0, until_max-1)] = len(targets) #the currently appended target index
                #print('currently added ',0, until_max-1)
            else:
                inp = torch.cat((inp, current_inp), dim=2)
                #inp.append(current_inp)
                frame_act_map[(act_tstamps[-1], act_tstamps[-1]+until_max-1)] = len(targets)
                act_tstamps.append(act_tstamps[-1]+ until_max)
                #print('currently adding ',act_tstamps[-1], act_tstamps[-1]+until_max)
                
            targets.append(current_target)

        is_action_recog = True # or training using the baseline with multiple actions and z is split (transformermulti)
        if is_action_recog and len(targets)< MAX_NUM_CLASSES:
            targets = np.asarray(targets)
            ntoadd = MAX_NUM_CLASSES - len(targets)
            padding = targets[-1] * np.ones(ntoadd, dtype=int)
            targets = np.concatenate((targets, padding), axis=0)
                
            #if current_inp is None:
        #print('****input size', inp.shape)
        return inp, targets, act_tstamps, frame_act_map


   

    def get_pose_data(self, annot, frame_ix):
        pose = self._load(annot, frame_ix)
        return pose

    def _load(self, vibe_data, frame_ix):
        pose_rep = self.pose_rep
        if pose_rep == "xyz" or self.translation:
            if getattr(self, "_load_joints3D", None) is not None:
                # Locate the root joint of initial pose at origin
                joints3D = self._load_joints3D(vibe_data, frame_ix)
                joints3D = joints3D - joints3D[0, 0, :]
                ret = to_torch(joints3D)
                if self.translation:
                    ret_tr = ret[:, 0, :]
            else:
                if pose_rep == "xyz":
                    raise ValueError("This representation is not possible.")
                if getattr(self, "_load_translation") is None:
                    raise ValueError("Can't extract translations.")
                ret_tr = self._load_translation(vibe_data, frame_ix)
                ret_tr = to_torch(ret_tr - ret_tr[0])

        if pose_rep != "xyz":
            if getattr(self, "_load_rotvec", None) is None:
                raise ValueError("This representation is not possible.")
            else:
                pose = self._load_rotvec(vibe_data, frame_ix)
                if not self.glob:
                    pose = pose[:, 1:, :]
                pose = to_torch(pose)
                if pose_rep == "rotvec":
                    ret = pose
                elif pose_rep == "rotmat":
                    ret = geometry.axis_angle_to_matrix(pose).view(*pose.shape[:2], 9)
                elif pose_rep == "rotquat":
                    ret = geometry.axis_angle_to_quaternion(pose)
                elif pose_rep == "rot6d":
                    ret = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))
        if pose_rep != "xyz" and self.translation:
            padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
            padded_tr[:, :3] = ret_tr
            ret = torch.cat((ret, padded_tr[:, None]), 1)
        ret = ret.permute(1, 2, 0).contiguous()
        return ret.float()

    def _load_joints3D(self, vibe_data, frame_ix):
        return vibe_data['joints3d'][frame_ix] #[begin_frame_ix:begin_frame_ix+self.num_frames, 0:24]

    def _load_rotvec(self, vibe_data, frame_ix):
        pose = vibe_data['pose'][frame_ix].reshape(-1, 24, 3)
        return pose

    

    def action_to_label(self, action):
        #todo
        #labels = []
        #for i, action in enumerate(actions):
        return self._action_to_label[action]
        
        labels = np.asarray(labels)
        return labels
    
    def label_to_action(self, label):
        import numbers
        if isinstance(label, numbers.Integral):
            #todo
            return self._label_to_action[label]
            #return '0'
        else:  # if it is one hot vector #todo not one hot, rather an array of tensors
            actions= []
            for l in label:
                actions.append(self._label_to_action[l.item()])
            return actions
            #label = np.argmax(label)
            #return self._label_to_action[label]
            

    #unused
    def get_label(self, action_annot, frame_ix):
        action = self.get_action(action_annot['actions'], frame_ix)
        act_letter_label = action['act']
        return self.action_to_label(act_letter_label)

    def parse_action(self, path, return_int=True):
        info = parse_info_name(path)["A"]
        if return_int:
            return int(info)
        else:
            return info
    #unused
    def get_action(self, action_data, frame_ix):
        actions = action_data.split(';')
        relevant_actions = []
        lower_same_act = min(frame_ix)
        higher_same_act = max(frame_ix)
        max_act_len = -1
        rel_ind = 0
        for action in actions:
            action_split = action.split(' ')
            act_label = action_split[0]
            act_start_frame = int(float(action_split[1]) * 30)
            act_end_frame = int(float(action_split[2]) * 30)
            if (min(frame_ix) >= act_start_frame and min(frame_ix) < act_end_frame) or max(frame_ix) <= act_end_frame or (act_start_frame >=min(frame_ix) and act_end_frame <=max(frame_ix)):
                if act_end_frame - act_start_frame > max_act_len:
                    rel_action = {'act_label': act_label, 'act_start_frame': act_start_frame, 'act_end_frame': act_end_frame}
                    relevant_actions.append(rel_action)
                    rel_ind = len(relevant_actions)-1
                    if act_start_frame > lower_same_act:
                        lower_same_act = act_start_frame
                    if act_end_frame < higher_same_act:
                        higher_same_act = act_end_frame
                    max_act_len = act_end_frame - act_start_frame

        frame_ix = frame_ix[(frame_ix >= lower_same_act) & (frame_ix <=higher_same_act) ]
        return relevant_actions[rel_ind]

    def action_to_action_name(self, action):
        return self._action_classes[action]

    def label_to_action_name(self, label):
        actions = self.label_to_action(label)
        actions_str = ''
        for act in actions:
            actions_str += self.action_to_action_name(act)
            actions_str += ','
        return actions_str

    
    
    def get_label_sample_ind(self, data_index):
        labels = [1]
        #while len(labels) < 2:
        if True:
            data_ind = [data_index]
            samples = [self.__getitem__(di) for di in data_ind]
            self.total_len += samples[0][0].shape[2]

            batch = collate(samples)
            x = batch["x"]
            mask = batch["mask"]
            lengths = mask.sum(1)
            labels = batch["y"]
            data_index = data_index + 1
        #print('got hold of one!!')
            

        if False:
            labels = np.random.randint(self.num_classes, size=(len(labels), labels.shape[1]) ) #todo, does not reflect a real sample, just for testing
            labels = torch.from_numpy(labels).to(batch['y'].device)
        act_timestamps = batch['action_timestamps'] if 'action_timestamps' in batch else None
        frame_act_map = batch['frame_act_map'] if 'frame_act_map' in batch else None
        return x, mask, lengths, labels, act_timestamps, frame_act_map

        

    def get_label_sample(self, label, n=1, return_labels=False, return_index=False):
        if self.split == 'train':
            index = self._train
        else:
            index = self._test

        #action = self.label_to_action(label)
        #print('*****action****', action, self._actions[index])
        #choices = np.argwhere(np.array(self._actions)[index] == action).squeeze(1)
        #choices = self._actions[index]
        #is_none = True
        #while is_none:
        if n == 1:
            y = -1
            if True: # while not (y==  label).all():
                data_index = np.random.randint(len(self._train)-1)
                current_data = self.data[data_index]
                x, y, _,_ = self._get_item_data_index(current_data)
                
                # data_index = index[np.random.choice(choices)]
                # id = list(self.data.keys())[data_index]
                # #vibe_pred_path = os.path.join(self.vibe_pred_path, id)
                # pkl_name = os.path.join(self.vibe_pred_path, '%s.pkl'%id)
                # vibe_data = joblib.load(pkl_name)
                # data_keys = list(vibe_data.keys())
                # for data_key in data_keys:
                #     if 'op_pose' in vibe_data[data_key]:
                #         break
                # action_annot = self.data[id]
                # x, y = self._get_item_data_index(vibe_data[data_key], action_annot)
                # if x is not None:
                #     is_none = False
                # #assert (label == y)
                # #y = label
        else:
            choices = [0]
            data_index = np.random.choice(choices, n)
            x = np.stack([self._get_item_data_index(index[di])[0] for di in data_index])
            y = label * np.ones(n, dtype=int)
        if return_labels:
            if return_index:
                return x, y, data_index
            return x, y
        else:
            if return_index:
                return x, data_index
            return x

    def __len__(self):
        num_seq_max = getattr(self, "num_seq_max", -1)
        if num_seq_max == -1:
            from math import inf
            num_seq_max = inf

        if self.split == 'train':
            #return 64
            return min(len(self._train), num_seq_max)
            #return 100
        else: #TODO
            return min(len(self._test), num_seq_max)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"

    def update_parameters(self, parameters):
        self.njoints, self.nfeats, _ = self[0][0].shape
        parameters["num_classes"] = self.num_classes
        parameters["nfeats"] = self.nfeats
        parameters["njoints"] = self.njoints


        #unused
    def get_mean_length_label(self, label):
        if self.num_frames != -1:
            return self.num_frames

        if self.split == 'train':
            index = self._train
        else:
            index = self._test

        action = self.label_to_action(label)
        choices = np.argwhere(self._actions[index] == action).squeeze(1)
        lengths = self._num_frames_in_video[np.array(index)[choices]]

        if self.max_len == -1:
            return np.mean(lengths)
        else:
            # make the lengths less than max_len
            lengths[lengths > self.max_len] = self.max_len
        return np.mean(lengths)
    #unused
    def get_stats(self):
        if self.split == 'train':
            index = self._train
        else:
            index = self._test

        numframes = self._num_frames_in_video[index]
        allmeans = np.array([self.get_mean_length_label(x) for x in range(self.num_classes)])

        stats = {"name": self.dataname,
                 "number of classes": self.num_classes,
                 "number of sequences": len(self),
                 "duration: min": int(numframes.min()),
                 "duration: max": int(numframes.max()),
                 "duration: mean": int(numframes.mean()),
                 "duration mean/action: min": int(allmeans.min()),
                 "duration mean/action: max": int(allmeans.max()),
                 "duration mean/action: mean": int(allmeans.mean())}
        return stats

    def split_train_test(self):
        act_dict = defaultdict(list)
        all_test_ids = []
        all_train_ids = []
        for id, item in enumerate(self.data):
            item['id'] = id
            act_dict[item['label']].append(item)
        for k,act_list in act_dict.items():
            len_test = max(1, len(act_list)//10)
            len_test = 0
            if len_test >= len(act_list) or len_test == 0:
                test_ids = []
            else:
                test_ids = np.random.randint(0, len(act_list)-1, len_test)
        
            for i in range(len(act_list)):
                if i in test_ids:
                    all_test_ids.append(act_list[i]['id'])
                else:
                    all_train_ids.append(act_list[i]['id'])
            
        return all_train_ids, all_test_ids


    def get_label_sample_batch(self, labels):
        samples = [self.get_label_sample(label, n=1, return_labels=True, return_index=False) for label in labels]
        batch = collate(samples)
        x = batch["x"]
        mask = batch["mask"]
        lengths = mask.sum(1)
        labels = batch["y"]
        return x, mask, lengths, labelsi