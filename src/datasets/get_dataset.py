def get_dataset(name="ntu13"):
    if name == "ntu13":
        from .ntu13 import NTU13
        return NTU13
    elif name == "uestc":
        from .uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .humanact12poses import HumanAct12Poses
        return HumanAct12Poses

    elif name.lower() == "charadesego" or name.lower() =='charades':
        from .charadesego import CharadesEgo
        return CharadesEgo
    elif name.lower() == 'charadesegomulti' or name.lower() == 'charadesmulti':
            from .charadesego_multi import CharadesEgoMulti
            return CharadesEgoMulti
    elif name.lower() == 'charadesegomultirecurrent' or name.lower() == 'charadesmultirecurrent' or name.lower() =='charadesrecurrent':
        from .charadesego_multi_recurrent import CharadesRecurrent
        return CharadesRecurrent
    
    elif name == "prox":
        from .prox import Prox
        return Prox
    
    elif name.lower() == "proxmulti" or name.lower == 'prox_multi' :
        from .prox_multi import ProxMulti
        return ProxMulti
    
    elif name.lower() == "proxrecurrent" or name.lower == 'prox_recurrent' :
        from .prox_recurrent import ProxRecurrent
        return ProxRecurrent

    elif name.lower() == "proxcluster" or name.lower == 'prox_cluster' :
        from .prox_cluster import ProxCluster
        return ProxCluster
   
   



def get_datasets(parameters):
    name = parameters["dataset"]

    DATA = get_dataset(name)
    dataset = DATA(split="train", **parameters)

    train = dataset

    # test: shallow copy (share the memory) but set the other indices
    from copy import copy
    #test = copy(train) #todo orig
    test = DATA(split="test", **parameters)
    test.split = test

    datasets = {"train": train,
                "test": DATA(split="test", **parameters)}

    # add specific parameters from the dataset loading
    dataset.update_parameters(parameters)

    return datasets
