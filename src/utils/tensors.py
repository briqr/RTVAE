import torch

def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    databatch = [b[0] for b in batch]
    labelbatch = [b[1] for b in batch]
    lenbatch = [len(b[0][0][0]) for b in batch]
    lenbatchTensor = torch.as_tensor(lenbatch)
    databatchTensor = collate_tensors(databatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor)
    if len(batch[0])>3:
        frameActTensors = [b[3] for b in batch]
    ret_batch = {"mask": maskbatchTensor, "lengths": lenbatchTensor}
    if len(batch[0]) == 4:  # indicates that this is the recurrent setting from the recurrent dataset which different labels for each subseq in the entire sequence batch element
        act_tstamps = [b[2] for b in batch]
        act_tstamps = [torch.as_tensor(a) for a in act_tstamps]
        act_tstamps = collate_tensors(act_tstamps)
        ret_batch['action_timestamps'] = act_tstamps
        labelbatchTensor = [torch.as_tensor(l) for l in labelbatch]
        labelbatchTensor = collate_tensors(labelbatchTensor)

        #databatchTensor = [collate_tensors(d) for d in databatch]
        #databatchTensor = collate_tensors(databatchTensor)

    else:
        labelbatchTensor = torch.as_tensor(labelbatch) #original model
        databatchTensor = collate_tensors(databatch)
    
    ret_batch["x"] = databatchTensor
    ret_batch["y"] = labelbatchTensor
    if len(batch[0])>3:
        ret_batch['frame_act_map'] = frameActTensors
    return ret_batch


def collate_tensors_multi(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_multi(batch):
    databatch = [b[0] for b in batch]
    labelbatch = [b[1] for b in batch]
    lenbatch = [len(b[0][0][0]) for b in batch]

    databatchTensor = collate_tensors_multi(databatch)
    labelbatchTensor = torch.as_tensor(labelbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)

    maskbatchTensor = lengths_to_mask(lenbatchTensor)
    batch = {"x": databatchTensor, "y": labelbatchTensor,
             "mask": maskbatchTensor, "lengths": lenbatchTensor}
    return batch
