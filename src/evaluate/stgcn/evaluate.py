import torch
import numpy as np
from .accuracy import calculate_accuracy
from .fid import calculate_fid
from .diversity import calculate_diversity_multimodality

from src.recognition.models.stgcn import STGCN


class Evaluation:
    def __init__(self, dataname, parameters, device, seed=None):
        layout = "smpl" if parameters["glob"] else "smpl_noglobal"
        if 'prox' in dataname:
            layout = 'smplx'
        model = STGCN(in_channels=parameters["nfeats"],
                      num_class=parameters["num_classes"],
                      graph_args={"layout": layout, "strategy": "spatial"},
                      edge_importance_weighting=True,
                      device=parameters["device"])

        model = model.to(parameters["device"])
        if 'uestc' in dataname:
            modelpath = "models/actionrecognition/uestc_rot6d_stgcn.tar"
        elif 'humanact12' in dataname:
            modelpath = 'models/actionrecognition/humanact12_gru.tar'
        elif 'prox' in dataname:
            #modelpath = 'recognition_proxrecurrent_multiact_120/checkpoint_0100.pth.tar'
            #modelpath ='recognition_proxrecurrent_multiact_frames60/checkpoint_0100.pth.tar'
            #modelpath = 'recognition_proxrecurrent_multiact/checkpoint_0100.pth.tar'
            #modelpath = 'recognition_proxrecurrent_max4_80/checkpoint_0100.pth.tar'
            #modelpath = 'recognition_proxrecurrent_multiact/checkpoint_0100.pth.tar'
            #modelpath = 'recognition_proxrecurrent_max1_80/checkpoint_0100.pth.tar'
            #modelpath = 'recognition_proxrecurrent_max1_60/checkpoint_0100.pth.tar'
            modelpath = 'recognition_proxrecurrent_max4_80/checkpoint_0100.pth.tar'

        elif 'charades' in dataname:
            modelpath = 'recognition_charadesrecurrent_max8_160/checkpoint_0100.pth.tar'
            modelpath = 'recognition_charadesrecurrent_max8_400/checkpoint_0100.pth.tar'
            modelpath = 'recognition_charadesrecurrent_max8_60_30actions/checkpoint_0100.pth.tar'
            #modelpath = 'recognition_proxrecurrent_max8_160/checkpoint_0100.pth.tar'
            #modelpath = 'recognition_charadesrecurrent_max4_200/checkpoint_0100.pth.tar'
            #modelpath = 'recognition_charadesrecurrent_max8_160/checkpoint_0100.pth.tar'
            #modelpath = 'recognition_charadesrecurrent_max4_120/checkpoint_0100.pth.tar'
            # modelpath = 'recognition_charadesrecurrent_max3_120/checkpoint_0100.pth.tar'
            # modelpath = 'recognition_charadesrecurrent_max2_80/checkpoint_0100.pth.tar'
            # modelpath = 'recognition_charadesrecurrent_max1_60/checkpoint_0100.pth.tar'


        print('***action recognition checkpoint %s' % modelpath)
        state_dict = torch.load(modelpath, map_location=parameters["device"])
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        self.num_classes = parameters["num_classes"]
        self.model = model

        self.dataname = dataname
        self.device = device

        self.seed = seed

    def compute_features(self, model, motionloader):
        # calculate_activations_labels function from action2motion
        activations = []
        labels = []
        with torch.no_grad():
            for idx, batch in enumerate(motionloader):
                activations.append(self.model(batch)["features"])
                labels.append(batch["y"])
            activations = torch.cat(activations, dim=0)
            labels = torch.cat(labels, dim=0)
        return activations, labels

    @staticmethod
    def calculate_activation_statistics(activations):
        activations = activations.cpu().numpy()
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def evaluate(self, model, loaders):
        def print_logs(metric, key):
            print(f"Computing stgcn {metric} on the {key} loader ...")

        metrics_all = {}
        for sets in ["train", "test"]: #todo:["train", "test"]:
            computedfeats = {}
            metrics = {}
            for key, loaderSets in loaders.items():
                loader = loaderSets[sets]

                metric = "accuracy"
                print_logs(metric, key)
                mkey = f"{metric}_{key}"
                metrics[mkey], _ = calculate_accuracy(model, loader,
                                                      self.num_classes,
                                                      self.model, self.device)
                print('mkey', mkey, metrics[mkey])
                # features for diversity
                
                print_logs("features", key)
                feats, labels = self.compute_features(model, loader)
                print_logs("stats", key)
                stats = self.calculate_activation_statistics(feats)

                computedfeats[key] = {"feats": feats,
                                    "labels": labels,
                                    "stats": stats}
                if True: #disable diversity calculation

                    print_logs("diversity", key)
                    ret = calculate_diversity_multimodality(feats, labels, self.num_classes,
                                                            seed=self.seed)
                    metrics[f"diversity_{key}"], metrics[f"multimodality_{key}"] = ret

            # taking the stats of the ground truth and remove it from the computed feats
            gtstats = computedfeats["gt"]["stats"]
            # computing fid
            for key, loader in computedfeats.items():
                metric = "fid"
                mkey = f"{metric}_{key}"

                stats = computedfeats[key]["stats"]
                try:
                    print('exception calculatung fid', mkey)
                    metrics[mkey] = float(calculate_fid(gtstats, stats))
                except:
                    metrics[mkey] = -1.0

            metrics_all[sets] = metrics

        metrics = {}
        for sets in ["train", "test"]: #todo["train", "test"]
            for key in metrics_all[sets]:
                metrics[f"{key}_{sets}"] = metrics_all[sets][key]
        return metrics
