from src.parser.evaluation import parser
import time


def main():
    parameters, folder, checkpointname, epoch, niter = parser()
    dataset = parameters["dataset"]
    print(dataset)
    if False: 
        epoch_range = [2000, 2100,100]
        #checkpoint = 'checkpoint_%s_pickup660.pth.tar'
        checkpoint = 'checkpoint_%s.pth.tar'
    else:
        epoch_range = [0, 1, 1]
    
    for epoch in range(epoch_range[0], epoch_range[1], epoch_range[2]):
        #checkpointname = checkpoint%str(epoch).zfill(4)
        if dataset in ["ntu13", "humanact12"]:
            start = time.time()
            from .gru_eval import evaluate
            evaluate(parameters, folder, checkpointname, epoch,  niter)
            end = time.time()
            print('time elapsed: ', end-start)
        elif dataset in ["uestc", "charadesego", "charadesrecurrent", "prox", 'proxmulti', 'proxrecurrent']:
            start = time.time()
            from src.evaluate.stgcn_eval import evaluate
            evaluate(parameters, folder, checkpointname, epoch, niter)
            end = time.time()
            print('time elapsed: ', end-start)
        else:
            raise NotImplementedError("This dataset is not supported.")
        


if __name__ == '__main__':
    main()
