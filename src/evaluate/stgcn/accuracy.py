import torch

def calculate_accuracy(model, motion_loader, num_labels, classifier, device):
    total_acc = 0
    num_actions = 0
    
    for batch in motion_loader:
        num_actions = len(batch['y'][0])
        break
    with torch.no_grad():
        for c in range(num_actions):
            confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
            for batch in motion_loader:
                #if torch.isnan(batch['x'].sum()):
                    #print('accuracy batchx is nan')
                    #continue
                batch_pred = classifier(batch)    
                pred_labels = batch_pred["yhat"][:,c*num_labels: (c+1)*num_labels].max(dim=1).indices
                for label, pred in zip(batch["y"][:, c], pred_labels):
                    confusion[label][pred] += 1
                if False:
                    print('**per class accuracy')
                    for i in range(confusion.shape[0]):
                        cl_acc = confusion[i,i]/confusion[i].sum() if confusion[i].sum()!=0 else 0
                        print('class %d' % i, ":", cl_acc)
            accuracy = torch.trace(confusion)/torch.sum(confusion)
            total_acc += accuracy.item()
        if num_actions == 0:
            return 0, 0
    return total_acc/num_actions, confusion




    # total_acc = 0
    #     for c in range(MAX_NUM_CLASSES):
    #         confusion = torch.zeros(self.num_class, self.num_class, dtype=int)
    #         if MAX_NUM_CLASSES > 1:
    #             yhat = batch["yhat"][:,c*self.num_class: (c+1)*self.num_class].max(dim=1).indices
    #             ygt = batch["y"][:,c]
    #         else:
    #             yhat = batch["yhat"].max(dim=1).indices
    #             ygt = batch["y"]
    #         for label, pred in zip(ygt, yhat):
    #             confusion[label][pred] += 1
    #         accuracy = torch.trace(confusion)/torch.sum(confusion)
    #         total_acc += accuracy
    #     return accuracy/MAX_NUM_CLASSES



# import torch


# def calculate_accuracy(model, motion_loader, num_labels, classifier, device):
#     confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
#     with torch.no_grad():
#         for batch in motion_loader:
#             batch_prob = classifier(batch)["yhat"]
#             batch_pred = batch_prob.max(dim=1).indices
#             for label, pred in zip(batch["y"], batch_pred):
#                 confusion[label][pred] += 1

#     accuracy = torch.trace(confusion)/torch.sum(confusion)
#     return accuracy.item(), confusion
