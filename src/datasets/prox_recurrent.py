#this file is changed in order for it to be similar to prox_multi with two actions when comparing with the baselie (returns two actions always and )
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

MAX_ACT_LEN = 120 #400 #full is trained with 80, linear is trained with 60
MAX_NUM_CLASSES = 5 #-1 indicates that the number of classes possible in a sample is ubnounded, 8 when evaluationg using the act recog model
class ProxRecurrent(Dataset):
    dataname = "proxrecurrent"
    
    def __init__(self, datapath="/home/ubuntu/datasets/PROX", split='train', **kargs):
        #print('*****dataset name %s****' %datapath)
        self.datapath = datapath 
        self.total_len = 0 
        #split='train'
        #split = 'test' #todo temp
        
        super().__init__(**kargs)
        class_file = os.path.join(datapath, 'annotations/action_labels.txt')
        prox_action_enumerator = dict()
        labels = []
        self.max_num_classes = MAX_NUM_CLASSES
        if MAX_ACT_LEN > 0:
            self.max_len = MAX_ACT_LEN
        else:
            self.max_len = kargs['MAX_ACT_LEN']
        action_indices = np.arange(58) #12) #48)
        #action_indices = [2,5,8,9,10,12,14,22,27,29,45] #subset 12#walking, sitting down, standing up, lying down, reading book
        #action_indices = [2, 5, 12, 13, 26, 29] #, 35, 45] #subset 6?
        #action_indices = [2, 5, 12, 13, 26, 29, 35, 45] #subset 8
        
        #print('**action indices', action_indices)
        total_num_actions = len(action_indices) #
        #print('****total num actions %d'% total_num_actions)
        #total_num_actions = 12
        action_indices = action_indices[:total_num_actions]
        #action_indices = np.arange(12)
        action_recog_indices = action_indices 
        #np.arange(10)
        self.class_map = init_class_map()

        with open (class_file, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if not idx in action_recog_indices:
                    continue
                label_actname = line.split(':')
                label = int(label_actname[0])
                if label not in self.class_map:
                    continue
                action = label_actname[1]
                label = self.class_map[label]
                prox_action_enumerator[label] = action
                labels.append(label)
        

        
        self._action_to_label = {x: i for i, x in enumerate(labels)}
        self._label_to_action = {i: x for i, x in enumerate(labels)}
        self._idx_to_action = {i: prox_action_enumerator[x] for i, x in enumerate(labels)}


        #print('action to label', self._idx_to_action)
        self.num_attempts = 0
        num_ego_videos = 0
        #self.smpl_pred_path = os.path.join(datapath, 'smpl_pred') #path for smpl and openpose predictions and indices of good sequences
        self.smpl_pred_path = os.path.join(datapath, 'smpl_processed') #path for smpl and openpose predictions and indices of good sequences
        self.annot_path = os.path.join(datapath, 'annotations')
             
        self.data_path = datapath
       


        self.num_classes = total_num_actions
        #self.num_classes = len(class_map) #todo disabled since I dont have time to retrain with the correct number of actions
        if split == 'test':
            data = pickle.load(open(os.path.join(datapath, 'processed_data_%s.pkl'%split), 'rb'))
        else:
            data = pickle.load(open(os.path.join(datapath, 'processed_data_%s.pkl'%split), 'rb'))

        #test_data = pickle.load(open(os.path.join(datapath, 'processed_data_test.pkl'%split), 'rb'))
       
       
        self.data = []
        self.test_data = []
        total_num_frames = 0
        prev_video = None
        current_max_cl = 0
        label_num_samples = defaultdict(lambda:0)
        for data_i, item in enumerate(data):
            #if data_i in self._test:
            #    continue
            #TODO
            if item['label'] in prox_action_enumerator:
                #item['label'] = np.random.randint(0,2)
                #print('length of sample %d: %d' %(data_i, len(item['body_pose'])))
                label_num_samples[item['label']] += 1
                total_num_frames += len(item['body_pose'])
                self.data.append(item)
                if prev_video is not None and prev_video == item['video_name']:
                    current_max_cl += 1
                else:
                    if current_max_cl > self.max_num_classes:
                        self.max_num_classes = current_max_cl
                        current_max_cl = 0
                prev_video = item['video_name']
        self.samples_weights = []
        for data_i, item in enumerate(self.data):
            weight_i = 1.0/label_num_samples[item['label']]
            self.samples_weights.append(weight_i)
        if MAX_NUM_CLASSES > 0:
            self.max_num_classes = MAX_NUM_CLASSES
        #print('maximum number of classes %d' %self.max_num_classes)
        
        train_file_name = os.path.join(kargs['folder'], 'prox_train_all.npy')
        test_file_name = os.path.join(kargs['folder'], 'prox_test_all.npy')
        if True: #not os.path.exists(train_file_name):
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
        
        # label_num_samples = defaultdict(lambda:0)
        # for ind in self._train:
        #     item = self.data[ind]
        #     label_num_samples[item['label']] += len(item['body_pose'])
        #print('number of samples per label')
        #for l in label_num_samples.keys():
        #    print(l, label_num_samples[l])

        
        #print('num of samples %d' %self.__len__())
        #self._train = range(len(self.data))

        #self._actions = np.random.randint(0, total_num_actions-1, self.__len__())


        self._action_classes = prox_action_enumerator

        self.id = 0

    
    def __getitem__(self, index):
        # the difference to prox_multi is that here we take all subsequences belonging to a video
        seq_data = []
        data_ind = self._train[index]
        if self.split == 'train':
            while True: 
                current_data = self.data[data_ind]
                seq_data.append(current_data)
                if False: #todo, experiment recursive using a single action
                    break
                if True and len(seq_data) == self.max_num_classes:
                    break
                if data_ind == len(self.data)-1 or self.data[data_ind]['video_name'] != self.data[data_ind+1]['video_name']:
                    break
                data_ind = (data_ind + 1)%len(self.data)

        else:
            data_ind = self._test[index]
                    
        if True:
            inp, target, action_tstamps, frame_act_map = self._get_item_data_index(seq_data)#(seq_data[0:1]) #seq_data[0:1] #todo, for debugging using 1 action (peract single action)
        # except:
        #     print('****exception in getitem')
        #     return self.__getitem__((index+1)%self.__len__())
        #print('act tstamps ', action_tstamps)

        return inp, target, action_tstamps, frame_act_map #, pose_conf, pose_dist



    def _get_item_data_index(self, annots):     
        #print('len pose', len(annot['pose']))
        inp = None
        #inp = []
        act_tstamps = []
        targets = []
        total_act_len = 0
        frame_act_map = RangeDict()
        for annot in annots:
            current_act_len = len(annot['body_pose'])

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

        is_action_recog = False # or training using the baseline with multiple actions and z is split and for training and evaluation CVAE
        if is_action_recog and len(targets)< MAX_NUM_CLASSES:
            targets = np.asarray(targets)
            ntoadd = MAX_NUM_CLASSES - len(targets)
            padding = targets[-1] * np.ones(ntoadd, dtype=int)
            targets = np.concatenate((targets, padding), axis=0)
                
            #if current_inp is None:
        #print('****input size', inp.shape)
        return inp, targets, act_tstamps, frame_act_map






    def _get_item_data_index_eval(self, annots):     
        #print('len pose', len(annot['pose']))
        inp = None
        #inp = []
        act_tstamps = []
        targets = []
        total_act_len = 0
        frame_act_map = RangeDict()
        for annot in annots:
            current_act_len = len(annot['body_pose'])

       
            total_act_len += current_act_len
            current_frame_ix = np.arange(0, current_act_len) 
            
            current_inp = self.get_pose_data(annot, current_frame_ix) #frame_ix) #
            current_target = self.action_to_label(annot['label'])

            #if len(inp) == 0:
            #print('inp.sum: ', current_inp.sum())
            if inp is None:
                #inp.append(current_inp)
                inp = current_inp
                act_tstamps.append(current_act_len)
                frame_act_map[(0, current_act_len-1)] = len(targets) #the currently appended target index
                #print('currently added ',0, until_max-1)
            else:
                inp = torch.cat((inp, current_inp), dim=2)
                #inp.append(current_inp)
                frame_act_map[(act_tstamps[-1], act_tstamps[-1]+current_act_len-1)] = len(targets)
                act_tstamps.append(act_tstamps[-1]+ current_act_len)
                #print('currently adding ',act_tstamps[-1], act_tstamps[-1]+until_max)
                
            targets.append(current_target)

        is_action_recog = False # or training using the baseline with multiple actions and z is split and for training and evaluation CVAE
        if is_action_recog and len(targets)< MAX_NUM_CLASSES:
            targets = np.asarray(targets)
            ntoadd = MAX_NUM_CLASSES - len(targets)
            padding = targets[-1] * np.ones(ntoadd, dtype=int)
            targets = np.concatenate((targets, padding), axis=0)
                
            #if current_inp is None:
        #print('****input size', inp.shape)
        return inp, targets, act_tstamps, frame_act_map



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
            actions= []
            for l in label:
                actions.append(self._label_to_action[l])
            return actions


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
        actions_str = ''

        for lab in label:
            if lab.item() in self.class_map:
                lab = self.class_map[lab.item()]
                act = self.label_to_action(lab)
                actions_str += self.action_to_action_name(act)
                actions_str += ','
        actions_str = actions_str[:-1]
        return actions_str




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
            #while(y!=label and iter < 5):
            if True:
                data_index = np.random.randint(len(self._train)-1)
                current_data = [self.data[data_index], self.data[data_index+1]]
                x, y, act_timestamps, frame_act_map = self._get_item_data_index_eval(current_data)
                iter += 1
                
                
        else:
            data_index = np.random.choice([1,2], n)
            x = np.stack([self._get_item_data_index(index[di])[0] for di in data_index])
            y = label * np.ones(n, dtype=int)
        if return_labels:
            if return_index:
                return x, y, data_index
            return x, y, act_timestamps, frame_act_map
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
            #return 4
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
        #samples = [self.get_label_sample(label, n=1, return_labels=True, return_index=False) for label in labels]
        data_index = np.random.randint(len(self._train)-1, size = len(labels))
        samples = [self.__getitem__(di) for di in data_index]
        batch = collate(samples)
        x = batch["x"]
        mask = batch["mask"]
        lengths = mask.sum(1)
        labels = batch["y"]

        if False:
            labels = np.random.randint(self.num_classes, size=(len(labels), labels.shape[1]) ) #todo, does not reflect a real sample, just for testing
            labels = torch.from_numpy(labels).to(batch['y'].device)
        act_timestamps = batch['action_timestamps'] if 'action_timestamps' in batch else None
        frame_act_map = batch['frame_act_map'] if 'frame_act_map' in batch else None
        return x, mask, lengths, labels, act_timestamps, frame_act_map


    def get_label_sample_ind(self, data_index):
        data_index = [data_index]
        samples = [self.__getitem__(di) for di in data_index]
        self.total_len += samples[0][0].shape[2]
 
        batch = collate(samples)
        x = batch["x"]
        mask = batch["mask"]
        lengths = mask.sum(1)
        labels = batch["y"]

        if False:
            labels = np.random.randint(self.num_classes, size=(len(labels), labels.shape[1]) ) #todo, does not reflect a real sample, just for testing
            labels = torch.from_numpy(labels).to(batch['y'].device)
        act_timestamps = batch['action_timestamps'] if 'action_timestamps' in batch else None
        frame_act_map = batch['frame_act_map'] if 'frame_act_map' in batch else None
        return x, mask, lengths, labels, act_timestamps, frame_act_map


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

def init_class_map():
    class_map = dict()
    class_map[0]=0
    class_map[1]=1
    class_map[2]=2
    for l in range(4,18):
        class_map[l]=l-1
    for l in range(4,18):
        class_map[l]=l-1
    for l in range(22,36):
        class_map[l]=l-5
    for l in range(22,36):
        class_map[l]=l-5
    for l in range(37,42):
        class_map[l]=l-6
    class_map[43]=36
    for l in range(45,58):
        class_map[l]=l-8
    return class_map

