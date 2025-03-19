import os
import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from src.gan.model.transformer_gan import TransformerGAN
from src.gan.trainer import train, test
from src.utils.tensors import collate
import src.utils.fixseed  # noqa

from src.parser.training import parser
from src.utils.get_model_and_data import get_model_and_data
from src.datasets.get_dataset import get_datasets
from src.gan.model.model_util import *


def do_epochs(model, datasets, parameters, optimizer, writer):
    dataset = datasets["train"]
    train_iterator = DataLoader(dataset, batch_size=parameters["batch_size"],
                                shuffle=True, num_workers=10, drop_last=True, collate_fn=collate)
    #dataset_val = datasets['test']
    # val_iterator = DataLoader(dataset_val, batch_size=len(dataset_val),
    #                             shuffle=False, num_workers=0, drop_last=True, collate_fn=collate)
    

    logpath = os.path.join(parameters["folder"], "training.log")
    with open(logpath, "w") as logfile:
        for epoch in range(1, parameters["num_epochs"]+1):
            print('**new epoch train ** %d' %epoch)
            dict_loss = train(model, optimizer, train_iterator, model.device)

            for key in dict_loss.keys():
                dict_loss[key] /= len(train_iterator)
                writer.add_scalar(f"Loss/{key}", dict_loss[key], epoch)

            epochlog = f"Epoch {epoch}, train losses: {dict_loss}"
            print(epochlog)
            print(epochlog, file=logfile)

            if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["num_epochs"]):
                checkpoint_path = os.path.join(parameters["folder"],
                                               'checkpoint_{:04d}.pth.tar'.format(epoch))
                print('Saving checkpoint {}'.format(checkpoint_path))
                torch.save(model.state_dict(), checkpoint_path)
            
            if False:#or epoch == 1 or epoch %10 == 0:
                #print('**new epoch val ** %d' %epoch)
                dict_loss = test(model, optimizer, val_iterator, model.device)
                for key in dict_loss.keys():
                    dict_loss[key] /= len(val_iterator)
                    writer.add_scalar(f"Val Loss/{key}", dict_loss[key], epoch)
                epochlog = f"Epoch {epoch}, val losses: {dict_loss}"
                print(epochlog)
                print(epochlog, file=logfile)

            
            writer.flush()






if __name__ == '__main__':
    # parse options
    parameters = parser()
    
    
    # logging tensorboard
    writer = SummaryWriter(log_dir=parameters["folder"])

    datasets = get_datasets(parameters)

    model = TransformerGAN()
    model.generator.apply(weights_init)



    optimizer = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=parameters["lr"])
   
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training model..")
    #do_epochs(model, datasets, parameters, optimizer, optimizer_d, optimizer_g, writer)
    do_epochs(model, datasets, parameters, optimizer, writer)

    writer.close()
