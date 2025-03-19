import torch
from tqdm import tqdm

from src.utils.fixseed import fixseed

from src.evaluate.stgcn.evaluate import Evaluation as STGCNEvaluation
# from src.evaluate.othermetrics.evaluation import Evaluation

from torch.utils.data import DataLoader
from src.utils.tensors import collate

import os

from .tools import save_metrics, format_metrics # for command line
#from tools import save_metrics, format_metrics # for VScode
from src.models.get_model import get_model as get_gen_model
from src.datasets.get_dataset import get_datasets
import src.utils.rotation_conversions as geometry
from src.evaluate_updated.tables.easy_table import *

def convert_x_to_rot6d(x, pose_rep):
    # convert rotation to rot6d
    if pose_rep == "rotvec":
        x = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(x))
    elif pose_rep == "rotmat":
        x = x.reshape(*x.shape[:-1], 3, 3)
        x = geometry.matrix_to_rotation_6d(x)
    elif pose_rep == "rotquat":
        x = geometry.matrix_to_rotation_6d(geometry.quaternion_to_matrix(x))
    elif pose_rep == "rot6d":
        x = x
    else:
        raise NotImplementedError("No geometry for this one.")
    return x


class NewDataloader:
    def __init__(self, mode, model, parameters, dataiterator, device):
        assert mode in ["gen", "rc", "gt"]
        
        pose_rep = parameters["pose_rep"]
        translation = parameters["translation"]

        self.batches = []

        with torch.no_grad():
            for databatch in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
                if mode == "gen":
                    classes = databatch["y"]
                    gendurations = databatch["lengths"]
                    act_tstamps = None
                    frame_act_map = None
                    if 'action_timestamps' in databatch:
                        act_tstamps = databatch['action_timestamps']
                        frame_act_map = databatch['frame_act_map']
                    batch = model.generate(classes, gendurations, action_timestamps=act_tstamps, frame_act_map=frame_act_map)
                    feats = "output"
                elif mode == "gt":
                    batch = dict()
                    for key, val in databatch.items():
                        if key!='action_timestamps' and key!='frame_act_map':
                            batch[key] = val.to(device)
                        else:
                            batch[key] = val
                    feats = "x"
                elif mode == "rc":
                    databatch = {key: val.to(device) for key, val in databatch.items()}
                    batch = model(databatch)
                    feats = "output"
                
             
                for key, val in batch.items():
                    if key!='action_timestamps' and key!='frame_act_map':
                        batch[key] = val.to(device)
                    else:
                        batch[key] = val

                if translation:
                    x = batch[feats][:, :-1]
                else:
                    x = batch[feats]

                x = x.permute(0, 3, 1, 2)
                x = convert_x_to_rot6d(x, pose_rep)
                x = x.permute(0, 2, 3, 1)

                batch["x"] = x

                self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)


def evaluate(parameters, folder, checkpointname, epoch, niter):
    torch.multiprocessing.set_sharing_strategy('file_system')

    bs = parameters["batch_size"]
    doing_recons = False

    device = parameters["device"]
    dataname = parameters["dataset"]

    # dummy => update parameters info
    # get_datasets(parameters)
    # faster: hardcode value for uestc


    compute_gt_gt = True
    if compute_gt_gt:
        datasetGT = {key: [get_datasets(parameters)[key],
                           get_datasets(parameters)[key]]
                     for key in ["train", "test"]}
    else:
        datasetGT = {key: [get_datasets(parameters)[key]]
                     for key in ["train", "test"]}

    print("Dataset loaded")


    #parameters["num_classes"] = datasetGT['train'][0].num_classes #31, 40
    
    parameters["nfeats"] = 6
    #parameters["num_classes"] = 58
    
    parameters["njoints"] = 25
    if 'prox' in parameters['dataset']:
        parameters['njoints'] = 23 #--no-translation 22, else 23
     



    model = get_gen_model(parameters)

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    model.outputxyz = False

    recogparameters = parameters.copy()
    recogparameters["pose_rep"] = "rot6d"
    recogparameters["nfeats"] = 6
    #recogparameters["num_classes"] = 50

    # Action2motionEvaluation
    stgcnevaluation = STGCNEvaluation(dataname, recogparameters, device)

    stgcn_metrics = {}
    # joints_metrics = {}
# pose_metrics = {}

    allseeds = list(range(niter))
    subsets = ['train', 'test'] #["train", "test"] #todo
    for seedi, seed in enumerate(allseeds):
        print('evaluation over the %d seed'%seedi)
        fixseed(seed)
        for key in ["train", "test"]:
            for data in datasetGT[key]:
                data.reset_shuffle()
                data.shuffle()

        dataiterator = {key: [DataLoader(data, batch_size=bs,
                                         shuffle=False, num_workers=8,
                                         collate_fn=collate, drop_last=False)
                              for data in datasetGT[key]]
                        for key in subsets}

        if doing_recons:
            reconsLoaders = {key: NewDataloader("rc", model, parameters,
                                                dataiterator[key][0],
                                                device)
                             for key in subsets}

        gtLoaders = {key: NewDataloader("gt", model, parameters,
                                        dataiterator[key][0],
                                        device)
                     for key in subsets}

        if compute_gt_gt:
            gtLoaders2 = {key: NewDataloader("gt", model, parameters,
                                             dataiterator[key][1],
                                             device)
                          for key in subsets}

        genLoaders = {key: NewDataloader("gen", model, parameters,
                                         dataiterator[key][0],
                                         device)
                      for key in subsets}

        loaders = {"gen": genLoaders,
                   "gt": gtLoaders}
        if doing_recons:
            loaders["recons"] = reconsLoaders

        if compute_gt_gt:
            loaders["gt2"] = gtLoaders2

        stgcn_metrics[seed] = stgcnevaluation.evaluate(model, loaders)
        del loaders

        # joints_metrics = evaluation.evaluate(model, loaders, xyz=True)
        # pose_metrics = evaluation.evaluate(model, loaders, xyz=False)

    metrics = {"feats": {key: [format_metrics(stgcn_metrics[seed])[key] for seed in allseeds] for key in stgcn_metrics[allseeds[0]]}}
    # "xyz": {key: [format_metrics(joints_metrics[seed])[key] for seed in allseeds] for key in joints_metrics[allseeds[0]]},
    # model.pose_rep: {key: [format_metrics(pose_metrics[seed])[key] for seed in allseeds] for key in pose_metrics[allseeds[0]]}}

    epoch = checkpointname.split("_")[1].split(".")[0]
    metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, metrics)
    folder, evaluation = os.path.split(evalpath)
    print_results(folder, evaluation)



    



# import torch
# from tqdm import tqdm

# from src.utils.fixseed import fixseed

# from src.evaluate.stgcn.evaluate import Evaluation as STGCNEvaluation
# # from src.evaluate.othermetrics.evaluation import Evaluation

# from torch.utils.data import DataLoader
# from src.utils.tensors import collate

# import os

# from .tools import save_metrics, format_metrics
# from src.models.get_model import get_model as get_gen_model
# from src.datasets.get_dataset import get_datasets
# import src.utils.rotation_conversions as geometry
# from src.evaluate_updated.tables.easy_table import *


# def convert_x_to_rot6d(x, pose_rep):
#     # convert rotation to rot6d
#     if pose_rep == "rotvec":
#         x = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(x))
#     elif pose_rep == "rotmat":
#         x = x.reshape(*x.shape[:-1], 3, 3)
#         x = geometry.matrix_to_rotation_6d(x)
#     elif pose_rep == "rotquat":
#         x = geometry.matrix_to_rotation_6d(geometry.quaternion_to_matrix(x))
#     elif pose_rep == "rot6d":
#         x = x
#     else:
#         raise NotImplementedError("No geometry for this one.")
#     return x


# class NewDataloader:
#     def __init__(self, mode, model, parameters, dataiterator, device):
#         assert mode in ["gen", "rc", "gt"]

#         pose_rep = parameters["pose_rep"]
#         translation = parameters["translation"]

#         self.batches = []

#         with torch.no_grad():
#             for databatch in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
#                 if mode == "gen":
#                     classes = databatch["y"]
#                     gendurations = databatch["lengths"]
#                     batch = model.generate(classes, gendurations)
#                     feats = "output"
#                 elif mode == "gt":
#                     batch = dict()
#                     for key, val in batch.items():
#                         if key!='action_timestamps' and key!='frame_act_map':
#                             batch[key] = val.to(device)
#                         else:
#                              batch[key] = val
                            
#                     feats = "x"
#                 elif mode == "rc":
#                     databatch = {key: val.to(device) for key, val in databatch.items()}
#                     batch = model(databatch)
#                     feats = "output"

#                 batch = {key: val.to(device) for key, val in batch.items() if val is not None}

#                 if translation:
#                     x = batch[feats][:, :-1]
#                 else:
#                     x = batch[feats]

#                 x = x.permute(0, 3, 1, 2)
#                 x = convert_x_to_rot6d(x, pose_rep)
#                 x = x.permute(0, 2, 3, 1)

#                 batch["x"] = x

#                 self.batches.append(batch)

#     def __iter__(self):
#         return iter(self.batches)


# def evaluate(parameters, folder, checkpointname, epoch, niter):
#     torch.multiprocessing.set_sharing_strategy('file_system')

#     bs = parameters["batch_size"]
#     doing_recons = False

#     device = parameters["device"]
#     dataname = parameters["dataset"]

#     # dummy => update parameters info
#     # get_datasets(parameters)
#     # faster: hardcode value for uestc

#     parameters["num_classes"] =48
#     parameters["nfeats"] = 6
#     parameters["njoints"] = 23 #todo #25, 23, 22

#     model = get_gen_model(parameters)

#     print("Restore weights..")
#     checkpointpath = os.path.join(folder, checkpointname)
#     state_dict = torch.load(checkpointpath, map_location=device)
#     model.load_state_dict(state_dict)
#     model.eval()

#     model.outputxyz = False

#     recogparameters = parameters.copy()
#     recogparameters["pose_rep"] = "rot6d"
#     recogparameters["nfeats"] = 6

#     # Action2motionEvaluation
#     stgcnevaluation = STGCNEvaluation(dataname, recogparameters, device)

#     stgcn_metrics = {}
#     # joints_metrics = {}
#     # pose_metrics = {}

#     compute_gt_gt = False
#     if compute_gt_gt:
#         datasetGT = {key: [get_datasets(parameters)[key],
#                            get_datasets(parameters)[key]]
#                      for key in ["train", "test"]}
#     else:
#         datasetGT = {key: [get_datasets(parameters)[key]]
#                      for key in ["train", "test"]}

#     print("Dataset loaded")

#     allseeds = list(range(niter))

#     for seed in allseeds:
#         fixseed(seed)
#         for key in ["train", "test"]:
#             for data in datasetGT[key]:
#                 data.reset_shuffle()
#                 data.shuffle()

#         dataiterator = {key: [DataLoader(data, batch_size=bs,
#                                          shuffle=False, num_workers=8,
#                                          collate_fn=collate)
#                               for data in datasetGT[key]]
#                         for key in ["train", "test"]}

#         if doing_recons:
#             reconsLoaders = {key: NewDataloader("rc", model, parameters,
#                                                 dataiterator[key][0],
#                                                 device)
#                              for key in ["train", "test"]}

#         gtLoaders = {key: NewDataloader("gt", model, parameters,
#                                         dataiterator[key][0],
#                                         device)
#                      for key in ["train", "test"]}

#         if compute_gt_gt:
#             gtLoaders2 = {key: NewDataloader("gt", model, parameters,
#                                              dataiterator[key][1],
#                                              device)
#                           for key in ["train", "test"]}

#         genLoaders = {key: NewDataloader("gen", model, parameters,
#                                          dataiterator[key][0],
#                                          device)
#                       for key in ["train", "test"]}

#         loaders = {"gen": genLoaders,
#                    "gt": gtLoaders}
#         if doing_recons:
#             loaders["recons"] = reconsLoaders

#         if compute_gt_gt:
#             loaders["gt2"] = gtLoaders2

#         stgcn_metrics[seed] = stgcnevaluation.evaluate(model, loaders)
#         del loaders

#         # joints_metrics = evaluation.evaluate(model, loaders, xyz=True)
#         # pose_metrics = evaluation.evaluate(model, loaders, xyz=False)

#     metrics = {"feats": {key: [format_metrics(stgcn_metrics[seed])[key] for seed in allseeds] for key in stgcn_metrics[allseeds[0]]}}
#     # "xyz": {key: [format_metrics(joints_metrics[seed])[key] for seed in allseeds] for key in joints_metrics[allseeds[0]]},
#     # model.pose_rep: {key: [format_metrics(pose_metrics[seed])[key] for seed in allseeds] for key in pose_metrics[allseeds[0]]}}

#     epoch = checkpointname.split("_")[1].split(".")[0]
#     metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

#     evalpath = os.path.join(folder, metricname)
#     print(f"Saving evaluation: {evalpath}")
#     save_metrics(evalpath, metrics)
#     folder, evaluation = os.path.split(evalpath)
#     print_results(folder, evaluation)
#     print(f"Saving evaluation: {evalpath}")
#     save_metrics(evalpath, metrics)
