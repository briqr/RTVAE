import torch


def calculate_accuracy(model, motion_loader, num_labels, classifier, device):
    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
    with torch.no_grad():
        for batch in motion_loader:
            batch_prob = classifier(batch)["yhat"]
            batch_pred = batch_prob.max(dim=1).indices
            for label, pred in zip(batch["y"], batch_pred):
                confusion[label][pred] += 1
    print('**per class accuracy')
    for i in range(confusion.shape[0]):
        cl_acc = float(confusion[i,i])/confusion[i].sum() if confusion[i].sum()!=0 else 0
        print('class %d' % i, ":", cl_acc)
    #print('***confusion matrix', confusion)
    accuracy = torch.trace(confusion)/torch.sum(confusion)
    return accuracy.item(), confusion
