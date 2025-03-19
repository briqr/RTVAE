import torch
from tqdm import tqdm
from torch.autograd import grad
import numpy as np
import torch.nn.functional as F
def train_or_test(model, optimizer, iterator, device, mode="train"):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    # loss of the epoch
    dict_loss = {loss: 0 for loss in model.losses}

    with grad_env():
        for i, batch_i in tqdm(enumerate(iterator), desc="Computing batch"):
            # Put everything in device
            batch = {key: val.to(device) for key, val in batch_i.items()}
            # batch = {key: val.to(device) for key, val in batch_i.items() if key!='x'}
            # if isinstance(batch_i['x'], tuple):
            #     batch['x'] =  batch_i['x'][0].to(device)
            # else:
            #     batch['x' ] = batch_i['x'].to(device)
            # batch_i = None


            if mode == "train":
                # update the gradients to zero
                optimizer.zero_grad()

            # forward pass
            batch = model(batch)
            mixed_loss, losses = model.compute_loss(batch)

            # 
            # real_valid2 = model.disc_vert(batch['x_xyz'])
            # fake_valid2 = model.disc_vert(batch['output'].detach())
            # gradient_penalty2 = compute_gradient_penalty(model.disc_vert, batch['x_xyz'], batch['output_xyz'].detach(), 1)
            # loss_dis2 = -torch.mean(real_valid2) + torch.mean(fake_valid2) + gradient_penalty2 * 10 / (1** 2)
            if False:
                optimizer_d.zero_grad()
                real_valid, real_feat  = model.disc(batch['x'], batch['y'], batch['mask'])
                fake_valid, _ = model.disc(batch['output'].detach(),  batch['y'])

                #if args.loss == 'hinge':
                #    loss_dis = torch.mean(nn.ReLU(inplace=True)(1.0 - real_valid)).to(device) + torch.mean(nn.ReLU(inplace=True)(1 + fake_valid)).to(device)
                
                gradient_penalty = compute_gradient_penalty(model.disc, batch['x'], batch['output'].detach(), batch['y'], 1)
                loss_dis1 = -torch.mean(real_valid) + torch.mean(fake_valid) + gradient_penalty * 10 # / (1** 2)
                print('loss terms ', real_valid.mean().item(), fake_valid.mean().item(), gradient_penalty.item())

                print('discriminator loss', loss_dis1.item())
                loss_dis = loss_dis1 #+ loss_dis2

                
    #            model.disc_vert.zero_grad()
                loss_dis.backward()
                optimizer_d.step()
                if i % 5 == 0:
                    optimizer_g.zero_grad()
                    fake_valid, fake_feat = model.disc(batch['output'],  batch['y'], batch['mask'])
                    fake_gen = model.generate_during_train(batch['y'])
                    fake_gen_score, _ = model.disc(fake_gen['output'], batch['y'])
                    gener_loss = -(torch.mean(fake_valid).to(device) + torch.mean(fake_gen_score)) + F.mse_loss(real_feat.detach(), fake_feat, reduction='mean')
                    #gener_loss = -(torch.mean(fake_valid).to(device)) + F.mse_loss(real_feat.detach(), fake_feat, reduction='mean')
                    print('generator loss', gener_loss.item())
                    gener_loss.backward(retain_graph=True)
                    optimizer_g.step()
            # score_right_params = model.disc(batch['x'], batch['mask']) #x_xyz
            # score_fake_param = model.disc(batch['output'].detach(), batch['mask']) #output_xyz

            # score_right_vert = model.disc_vert(batch['x_xyz'], batch['mask']) #x_xyz
            # score_fake_vert = model.disc_vert(batch['output_xyz'].detach(), batch['mask']) #output_xyz

            # bsz = batch['x'].shape[0]
            # epsilon = torch.rand(bsz).to(device)[:,None] #, None, None]

            # smpl_sample = epsilon[:, None, None]* batch['x'] + (1 - epsilon[:, None, None]) * batch['output'].detach()
            # smpl_sample = smpl_sample.flatten(1)
            # smpl_sample.requires_grad = True
            # score_sample_params = model.disc(smpl_sample, batch['mask'])
            
            
            # vert_sample = epsilon[:, None, None] * batch['x_xyz'].detach() + (1 - epsilon[:, None, None]) * batch['output_xyz'].detach()
            # vert_sample = vert_sample.flatten(1)
            # vert_sample.requires_grad = True
            # score_sample_vert = model.disc_vert(vert_sample, batch['mask'])

            # gradient_s = grad(score_sample_params, smpl_sample, torch.ones_like(score_sample_params),
            #                                 create_graph=True)[0].mean(1)

            # gradient_v = grad(score_sample_vert, vert_sample, torch.ones_like(score_sample_vert),
            #                                 create_graph=True)[0].mean(1)

            # gradient_norm1 = gradient_s
            # gradient_norm2 = gradient_v



            # lamb = 10
            # alpha = 1
            # loss_d_p = (score_fake_param  - (1 + alpha) * score_right_params + lamb * (
            #         torch.max(torch.tensor(0, dtype=torch.float32, device=device), gradient_norm1 - 1).pow(2))).mean()
            
            # loss_d_v = (score_fake_vert - (1 + alpha) * score_right_vert+ lamb * (
            #         torch.max(torch.tensor(0, dtype=torch.float32, device=device), gradient_norm2 - 1).pow(2))).mean()
            # loss_d = loss_d_p + loss_d_v
            # loss_d.backward()
            # optimizer_d.step()
            
            # optimizer_g.zero_grad()
            # score_fake_param = model.disc(batch['output'], batch['mask'])
            # loss_g = -(score_fake_param).mean()
            # loss_g.backward(retain_graph=True)
            # optimizer_g.step()



            
            for key in dict_loss.keys():
                #if 'rc' in key:
                #    continue
                dict_loss[key] += losses[key]

            if mode == "train":
                # backward pass
                #if mixed_loss > 1000:
                #    print('skipped loss, !!!!!!!!!!!!!!!!batch %d ' %i)
                #   continue
                #model.parameters()
                mixed_loss.backward()
                # update the weights
                optimizer.step()
    return dict_loss


def compute_gradient_penalty(D, real_samples, fake_samples, y, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates, y)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty

def train(model, optimizer, iterator, device):
     return train_or_test(model, optimizer, iterator, device, mode="train")


def test(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, iterator, device, mode="test")
