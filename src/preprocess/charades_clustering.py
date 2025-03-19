from calendar import c
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from src.train.trainer import train
from src.utils.tensors import collate
import src.utils.fixseed  # noqa

from src.parser.training import parser
from src.utils.get_model_and_data import get_model_and_data
from tqdm import tqdm
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CLUSTER_DISTANCE_THRESH = 0.3

def find_keyframes(datasets):
    dataset = datasets["train"]
    iterator = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate)
    for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
        # Put everything in device
        batch = {key: val for key, val in batch.items()}

        clusters = batch['x'][0].to(device)
        clusters = clusters.permute(2, 0, 1)

        cluster_indices = torch.range(0, clusters.shape[0]-1).int()
        cluster_indices = [torch.tensor(i).view(1) for i in cluster_indices]
        cluster_distances = compute_distance(clusters)
        keep_clustering = True
        while(keep_clustering):
            #keep_clustering = cluster_min(cluster_indices, cluster_distances)
            keep_clustering = cluster_average(clusters, cluster_indices, cluster_distances)
            
        a = 0
 



def cluster_average(clusters, cluster_indices, cluster_distances):
    min_dist_ind = torch.argmin(cluster_distances)
    min_dist = torch.min(cluster_distances)
    print('min dist', min_dist.item())
    if min_dist > CLUSTER_DISTANCE_THRESH:
        return False
    i, j = min_dist_ind//cluster_distances.shape[0], min_dist_ind%cluster_distances.shape[1]
    cluster_indices[i] = torch.cat((cluster_indices[i], cluster_indices[j]))
    cluster_indices[j] = torch.tensor(-1).view(1)
    print('distance sum before', cluster_distances[cluster_distances<100000].sum())
    update_distance_avg(clusters, cluster_distances, cluster_indices, i)
    print('distance sum after', cluster_distances[cluster_distances<100000].sum())

    cluster_distances[i,i] = float('inf')
    cluster_distances[:,j] = float('inf')
    cluster_distances[j,:] = float('inf')
    return True




def update_distance_avg(clusters, cluster_dist, cluster_indices, updated_cluster_index):
    updated_center = torch.mean(clusters[cluster_indices[updated_cluster_index].long()], dim=0)
    for j in range(cluster_dist.shape[1]):
        if len(cluster_indices[j]) == 1 and cluster_indices[j]==-1:
            continue
        current_center = torch.mean(clusters[cluster_indices[j].long()], dim=0)
        cluster_dist[updated_cluster_index, j] = torch.norm(current_center-updated_center)
    # n = clusters.shape[0]
    # num_joints = clusters.shape[1]
    # clusters1 = clusters.unsqueeze(1).expand(-1,n, -1, -1)
    # clusters2 = clusters.unsqueeze(0).expand(n,-1, -1, -1)
    # clusters2 = clusters2.contiguous().view(n**2, num_joints* clusters.shape[2])
    # clusters1 = clusters1.contiguous().view(n**2, num_joints* clusters.shape[2])
    # dist = torch.norm(clusters1-clusters2,dim=-1)
    # dist = dist.reshape(n, n)
    # for i in range(n):
    #     dist[i,i] = torch.tensor(float('inf')).to(device)
   
    # return dist



def cluster_min(cluster_indices, cluster_distances):
    min_dist_ind = torch.argmin(cluster_distances)
    min_dist = torch.min(cluster_distances)
    print('min dist', min_dist.item())
    if min_dist > CLUSTER_DISTANCE_THRESH:
        return False
    i,j = min_dist_ind//cluster_distances.shape[0], min_dist_ind%cluster_distances.shape[1]
    cluster_indices[i] = torch.cat((cluster_indices[i], cluster_indices[j]))
    cluster_indices[j] = torch.tensor(-1).view(1)

    #cluster_indices[i].cat(j)
    #torch.cat([torch.tensor(cluster_indices[i]).view(1).to(device), torch.tensor(j).view(1).to(device)])
    cluster_distances[i,:] = torch.min(cluster_distances[i, :], cluster_distances[j,:]) 
    cluster_distances[i,i] = float('inf')
    cluster_distances[:,j] = float('inf')
    cluster_distances[j,:] = float('inf')
    return True

def compute_distance(clusters):
    n = clusters.shape[0]
    num_joints = clusters.shape[1]
    clusters1 = clusters.unsqueeze(1).expand(-1,n, -1, -1)
    clusters2 = clusters.unsqueeze(0).expand(n,-1, -1, -1)
    clusters2 = clusters2.contiguous().view(n**2, num_joints* clusters.shape[2])
    clusters1 = clusters1.contiguous().view(n**2, num_joints* clusters.shape[2])
    dist = torch.norm(clusters1-clusters2,dim=-1)
    dist = dist.reshape(n, n)
    for i in range(n):
        dist[i,i] = torch.tensor(float('inf')).to(device)
   
    return dist
if __name__ == '__main__':
    # parse options
    parameters = parser()

    model, datasets = get_model_and_data(parameters)
    model = None

    find_keyframes(datasets)
