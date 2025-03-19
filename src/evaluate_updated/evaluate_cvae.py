from src.parser.evaluation import parser
import src.utils.fixseed  # noqa

def main():
    parameters, folder, checkpointname, epoch, niter = parser()

    dataset = parameters["dataset"]
    print(dataset)
    if dataset in ["ntu13", "humanact12"]:
        from .gru_eval import evaluate
        evaluate(parameters, folder, checkpointname, epoch, niter)
    elif dataset in ["uestc", "charadesego", "prox", 'proxmulti', 'proxrecurrent']:
        from src.evaluate_updated.stgcn_eval import evaluate
        evaluate(parameters, folder, checkpointname, epoch, niter)
    else:
        raise NotImplementedError("This dataset is not supported.")


if __name__ == '__main__':
    main()
