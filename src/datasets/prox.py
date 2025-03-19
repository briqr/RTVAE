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

class Prox(Dataset):
    dataname = "prox"
    

    def __init__(self, datapath="/home/ubuntu/datasets/PROX", split='train', **kargs):
        print('*****dataset name %s****' %datapath)
        self.datapath = datapath  
        super().__init__(**kargs)
        class_file = os.path.join(datapath, 'annotations/action_labels.txt')
        prox_action_enumerator = dict()
        labels = []
        self.max_num_classes = 1
        #action_indices = [0, 1, 4, 10, 22, 27, 28] 
        #action_indices = [0, 1, 27, 28] #raising hand, turning around, drumming
        #action_indices = [2,5, 12, 22] #,8,9, 10, 14, 27, 29, 45] #walking, sitting down, standing up, lying down, reading book
        action_indices = [5, 12]
        action_indices = np.arange(48)
        print('**action indices', action_indices)
        total_num_actions = len(action_indices) #
        print('****total num actions %d'% total_num_actions)
        #total_num_actions = 12
        action_indices = action_indices[:total_num_actions]
        #action_indices = np.arange(12)
        with open (class_file, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if not idx in action_indices:
                    continue
                label_actname = line.split(':')
                label = int(label_actname[0])
                action = label_actname[1]
                prox_action_enumerator[label] = action
                labels.append(label)
        
        self._action_to_label = {x: i for i, x in enumerate(labels)}
        self._label_to_action = {i: x for i, x in enumerate(labels)}
        self._idx_to_action = {i: prox_action_enumerator[x] for i, x in enumerate(labels)}


        print('action to label', self._idx_to_action)
        self.num_attempts = 0
        num_ego_videos = 0
        #self.smpl_pred_path = os.path.join(datapath, 'smpl_pred') #path for smpl and openpose predictions and indices of good sequences
        self.smpl_pred_path = os.path.join(datapath, 'smpl_processed') #path for smpl and openpose predictions and indices of good sequences
        self.annot_path = os.path.join(datapath, 'annotations')
             
        self.data_path = datapath
       


        self.num_classes = total_num_actions
        data = pickle.load(open(os.path.join(datapath, 'processed_data.pkl'), 'rb'))
       
       
        self.data = []
        total_num_frames = 0
        for data_i, item in enumerate(data):
            #if data_i in self._test:
            #    continue
            #TODO
            if item['label'] in prox_action_enumerator:
                #item['label'] = np.random.randint(0,2)
                #print('length of sample %d: %d' %(data_i, len(item['body_pose'])))
                total_num_frames += len(item['body_pose'])
                self.data.append(item)
        
        train_file_name = os.path.join(kargs['folder'], 'prox_train_all.npy')
        test_file_name = os.path.join(kargs['folder'], 'prox_test_all.npy')
        if not os.path.exists(train_file_name):
            self._train, self._test = self.split_train_test()
            np.save(train_file_name, self._train)
            np.save(test_file_name, self._test)
            #self._test = np.random.randint(0, len(self.data)-1, size=5)

            #val_ind = np.random.randint(0,len(self._test)-1, size=5)
            #self._val = self._test[val_ind]
            #np.save('val', self._val)
        else:
            self._test = np.load(test_file_name)
            self._train = np.load(train_file_name)
            self._val = self._test
            
        # all = list(range(len(self.data)))
        # self._train = list(set(all) - set(self._test))
        
        label_num_samples = defaultdict(lambda:0)
        for ind in self._train:
            item = self.data[ind]
            label_num_samples[item['label']] += len(item['body_pose'])
        #print('number of samples per label')
        #for l in label_num_samples.keys():
        #    print(l, label_num_samples[l])

        
        print('num of samples %d' %self.__len__())
        #self._train = range(len(self.data))

        #self._actions = np.random.randint(0, total_num_actions-1, self.__len__())


        self._action_classes = prox_action_enumerator

        self.id = 0

    
    def __getitem__(self, index):

        if self.split == 'train':
            data_ind = self._train[index]
        else:
            data_ind = self._test[index]
            
        current_data = self.data[data_ind]

        try:
            inp, target, pose_conf, pose_dist = self._get_item_data_index(current_data)
        except:
            print('****exception in getitem')
            return self.__getitem__((index+1)%self.__len__())
        if inp is None or target is None: #shouldn't happen anymore
            print('could not find a long enough subsequence with action annotations')
            return self.__getitem__((index+1)%self.__len__())
        #print('id', self.id, 'index', index, current_data['video_name'], current_data['seq_ind'], current_data['label'])
        self.id += 1

        return inp, target#, pose_conf, pose_dist



    def _get_item_data_index(self, annot):     
        #print('len pose', len(annot['pose']))
        begin_frame_ix = 0
        nframes = len(annot['body_pose']) #self._num_frames_in_video[data_index]
        if True and nframes > self.num_frames: 
            begin_frame_ix = np.random.randint(0, nframes-self.num_frames)
        if nframes >= self.num_frames:
            frame_ix = np.arange(begin_frame_ix, begin_frame_ix+self.num_frames)
        else:
            #nframes = len(annot['pose']) #self._num_frames_in_video[data_index]
            if self.num_frames == -1 and (self.max_len == -1 or nframes <= self.max_len):
                frame_ix = np.arange(nframes)
            else:
                if self.num_frames == -2:
                    if self.min_len <= 0:
                        raise ValueError("You should put a min_len > 0 for num_frames == -2 mode")
                    if self.max_len != -1:
                        max_frame = min(nframes, self.max_len)
                    else:
                        max_frame = nframes

                    num_frames = random.randint(self.min_len, max(max_frame, self.min_len))
                else:
                    num_frames = self.num_frames if self.num_frames != -1 else self.max_len
                # sampling goal: input: ----------- 11 nframes
                #                       o--o--o--o- 4  ninputs
                #
                # step number is computed like that: [(11-1)/(4-1)] = 3
                #                   [---][---][---][-
                # So step = 3, and we take 0 to step*ninputs+1 with steps
                #                   [o--][o--][o--][o-]
                # then we can randomly shift the vector
                #                   -[o--][o--][o--]o
                # If there are too much frames required
                if num_frames > nframes:
                    fair = False  # True
                    if fair:
                        # distills redundancy everywhere
                        choices = np.random.choice(range(nframes),
                                                num_frames,
                                                replace=True)
                        frame_ix = sorted(choices)
                    else:
                        # adding the last frame until done
                        ntoadd = max(0, num_frames - nframes)
                        lastframe = nframes - 1
                        padding = lastframe * np.ones(ntoadd, dtype=int)
                        frame_ix = np.concatenate((np.arange(0, nframes),
                                                padding))

                elif self.sampling in ["conseq", "random_conseq"]:
                    step_max = (nframes - 1) // (num_frames - 1)
                    if self.sampling == "conseq":
                        if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                            step = step_max
                        else:
                            step = self.sampling_step
                    elif self.sampling == "random_conseq":
                        step = random.randint(1, step_max)

                    lastone = step * (num_frames - 1)
                    shift_max = nframes - lastone - 1
                    shift = random.randint(0, max(0, shift_max - 1))
                    frame_ix = shift + np.arange(0, lastone + 1, step)

                elif self.sampling == "random":
                    choices = np.random.choice(range(nframes),
                                            num_frames,
                                            replace=False)
                    frame_ix = sorted(choices)

                else:
                    raise ValueError("Sampling not recognized.")


        inp = self.get_pose_data(annot, frame_ix) #frame_ix) #
        if inp is None:
            return None, None, None, None
        target = self.action_to_label(annot['label'])
        op_conf = None
        op_dist = None
        return inp, target, op_conf, op_dist


    def get_pose_data(self, annot, frame_ix):
        #label = self.get_label(action_annot, frame_ix)
        pose = self._load(annot, frame_ix)
        return pose

    def _load(self, smpl_data, frame_ix):
        pose_rep = self.pose_rep
        if pose_rep == "xyz" or self.translation:
            if getattr(self, "_load_joints3D", None) is not None:
                # Locate the root joint of initial pose at origin
                joints3D = self._load_joints3D(smpl_data, frame_ix)
                joints3D = joints3D - joints3D[0, 0, :]
                ret = to_torch(joints3D)
                if self.translation:
                    ret_tr = ret[:, 0, :]
            else:
                if pose_rep == "xyz":
                    raise ValueError("This representation is not possible.")
                if getattr(self, "_load_translation") is None:
                    raise ValueError("Can't extract translations.")
                ret_tr = self._load_translation(smpl_data, frame_ix)#[:,0,:]
                if ret_tr is None:
                    print('translation is None!, returning None')
                    return None
                else:
                    ret_tr = ret_tr[:,0,:]
                    
                ret_tr = to_torch(ret_tr - ret_tr[0])

        if pose_rep != "xyz":
            if getattr(self, "_load_axisangle_vec", None) is None:
                raise ValueError("This representation is not possible.")
            else:
                pose = self._load_axisangle_vec(smpl_data, frame_ix)
                #pose = geometry.axis_angle_to_matrix(pose.reshape(pose.shape[0],-1,3))
                #pose = pose.view(*pose.shape[:2], 9)
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
                    ret = geometry.axis_angle_to_matrix(pose.reshape(pose.shape[0],-1,3))
                    ret = geometry.matrix_to_rotation_6d(ret)
        if pose_rep != "xyz" and self.translation:
            padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
            padded_tr[:, :3] = ret_tr[:,]
            ret = torch.cat((ret, padded_tr[:, None]), 1)
        ret = ret.permute(1, 2, 0).contiguous()
        return ret.float()

   
    def _load_axisangle_vec(self, smpl_data, frame_ix):
        body_pose = smpl_data['body_pose'][frame_ix]
        #print('sum, video_name', smpl_data['video_name'], smpl_data['seq_ind'], smpl_data['sample_id'], smpl_data['body_pose'][frame_ix].sum())
        body_pose = body_pose.reshape(-1, 63)

        global_orient = smpl_data['global_orient'][frame_ix]
        global_orient = global_orient.reshape(-1, 3)

        #hand_pose = torch.zeros(body_pose.shape[0],6).to(body_pose.device)
        #body_pose = torch.cat([global_orient, body_pose, hand_pose], dim=1)#in case I want to experiment with rendering using the SMPL model and use the default hand pose

        body_pose = torch.cat([global_orient, body_pose], dim=1)
        return body_pose

    def _load_translation(self, smpl_data, frame_ix):
        if len(smpl_data['transl']) < 1:
            return None
        try:
            transl = smpl_data['transl'][frame_ix]
        except:
            print('exception obtaining translation!!!!!!')
            a = 0
        return transl

    

    def action_to_label(self, action):
        #todo
        return self._action_to_label[action]

    def label_to_action(self, label):
        import numbers
        if isinstance(label, numbers.Integral):
            #todo
            return self._label_to_action[label]
            #return '0'
        else:  # if it is one hot vector
            #label = np.argmax(label)
            return self._label_to_action[label.item()]
            #todo
            #return '0'


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
        action = self.label_to_action(label)
        return self.action_to_action_name(action)


    def get_label_sample_ind(self, ind):
        item = [self.__getitem__(ind)]
        batch = collate(item)
        x = batch["x"]
        mask = batch["mask"]
        lengths = mask.sum(1)
        labels = batch["y"]
        return x, mask, lengths, labels

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
            iter = 0
            while(y!=label and iter < 5):
                data_index = np.random.randint(len(self._train)-1)
                current_data = self.data[data_index]
                x, y, _,_ = self._get_item_data_index(current_data)
                iter += 1
                
                
        else:
            data_index = np.random.choice([1,2], n)
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
            #train_files = os.listdir(self.vibe_pred_path)
            return min(len(self._train), num_seq_max)
            #return 10
        else: #TODO
            return min(len(self._test), num_seq_max)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"

   
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


    def get_label_sample_batch(self, labels):
        samples = [self.get_label_sample(label, n=1, return_labels=True, return_index=False) for label in labels]
        batch = collate(samples)
        x = batch["x"]
        mask = batch["mask"]
        lengths = mask.sum(1)
        labels = batch["y"]
        return x, mask, lengths, labels


    def split_train_test(self):
        act_dict = defaultdict(list)
        all_test_ids = []
        all_train_ids = []
        for id, item in enumerate(self.data):
            item['id'] = id
            act_dict[item['label']].append(item)
        for k,act_list in act_dict.items():
            len_test = max(1, len(act_list)//10)
            if len_test >= len(act_list):
                test_ids = []
            else:
                test_ids = np.random.randint(0, len(act_list)-1, len_test)
        
            for i in range(len(act_list)):
                if i in test_ids:
                    all_test_ids.append(act_list[i]['id'])
                else:
                    all_train_ids.append(act_list[i]['id'])
            
        return all_train_ids, all_test_ids