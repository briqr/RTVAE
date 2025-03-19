from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid, save_image

from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy


from utils import *
from src.models.modeltype.TransGAN import *
from src.parser.training import parser
from src.utils.get_model_and_data import get_model_and_data
from torch.utils.data import DataLoader
from src.datasets.get_dataset import get_datasets
from src.utils.tensors import collate

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default= 32 , help='Size of image for discriminator input.')
parser.add_argument('--initial_size', type=int, default=8 , help='Initial size for generator.')
parser.add_argument('--patch_size', type=int, default=4 , help='Patch size for generated image.')
parser.add_argument('--num_classes', type=int, default=1 , help='Number of classes for discriminator.')
parser.add_argument('--lr_gen', type=float, default=0.0001 , help='Learning rate for generator.')
parser.add_argument('--lr_dis', type=float, default=0.0001 , help='Learning rate for discriminator.')
parser.add_argument('--weight_decay', type=float, default=1e-3 , help='Weight decay.')
parser.add_argument('--latent_dim', type=int, default=512 , help='Latent dimension.')
parser.add_argument('--n_critic', type=int, default=5 , help='n_critic.')
parser.add_argument('--max_iter', type=int, default=500000 , help='max_iter.')
parser.add_argument('--gener_batch_size', type=int, default=64 , help='Batch size for generator.')
parser.add_argument('--dis_batch_size', type=int, default=32 , help='Batch size for discriminator.')
parser.add_argument('--epoch', type=int, default=200 , help='Number of epoch.')
parser.add_argument('--output_dir', type=str, default='checkpoint' , help='Checkpoint.')
parser.add_argument('--dim', type=int, default=384 , help='Embedding dimension.')
parser.add_argument('--img_name', type=str, default="img_name" , help='Name of pictures file.')
parser.add_argument('--optim', type=str, default="Adam" , help='Choose your optimizer')
parser.add_argument('--loss', type=str, default="wgangp_eps" , help='Loss function')
parser.add_argument('--phi', type=int, default="1" , help='phi')
parser.add_argument('--beta1', type=int, default="0" , help='beta1')
parser.add_argument('--beta2', type=float, default="0.99" , help='beta2')
parser.add_argument('--lr_decay', type=str, default=True , help='lr_decay')
parser.add_argument('--diff_aug', type=str, default="translation,cutout,color", help='Data Augmentation')

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"



device = torch.device(dev)
print("Device:",device)

args = parser.parse_args()

input_dim = 25*6
patch_size = 64
heads = 3
init_size = 8
generator= Generator(depth1=5, depth2=4, depth3=2, initial_size=init_size, dim=input_dim, heads=heads, mlp_ratio=4, drop_rate=0.1)#,device = device)
generator.to(device)

discriminator = Discriminator(diff_aug = args.diff_aug, image_size=32, patch_size=patch_size, input_channel=3, num_classes=1,
                 dim=input_dim, depth=7, heads=heads, mlp_ratio=4,
                 drop_rate=0., num_frames=patch_size-1)
discriminator.to(device)


generator.apply(inits_weight)
discriminator.apply(inits_weight)

if args.optim == 'Adam':
    optim_gen = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_gen, betas=(args.beta1, args.beta2))

    optim_dis = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),lr=args.lr_dis, betas=(args.beta1, args.beta2))
    
elif args.optim == 'SGD':
    optim_gen = optim.SGD(filter(lambda p: p.requires_grad, generator.parameters()),
                lr=args.lr_gen, momentum=0.9)

    optim_dis = optim.SGD(filter(lambda p: p.requires_grad, discriminator.parameters()),
                lr=args.lr_dis, momentum=0.9)

elif args.optim == 'RMSprop':
    optim_gen = optim.RMSprop(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_dis, eps=1e-08, weight_decay=args.weight_decay, momentum=0, centered=False)

    optim_dis = optim.RMSprop(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_dis, eps=1e-08, weight_decay=args.weight_decay, momentum=0, centered=False)

gen_scheduler = LinearLrDecay(optim_gen, args.lr_gen, 0.0, 0, args.max_iter * args.n_critic)
dis_scheduler = LinearLrDecay(optim_dis, args.lr_dis, 0.0, 0, args.max_iter * args.n_critic)


print("optim:",args.optim)

fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'

writer=SummaryWriter()
writer_dict = {'writer':writer}
writer_dict["train_global_steps"]=0
writer_dict["valid_global_steps"]=0

def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
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


def train(generator, discriminator, train_loader, optim_gen, optim_dis,
        epoch, writer, schedulers, img_size=32, latent_dim = args.latent_dim,
        n_critic = args.n_critic,
        gener_batch_size=args.gener_batch_size, device="cuda:0"):


    writer = writer_dict['writer']
    gen_step = 0

    generator = generator.train()
    discriminator = discriminator.train()

    #transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    
    
    # logging tensorboard

    print('start of epoch %d' %epoch)
    for index, batch in enumerate(train_loader):

        global_steps = writer_dict['train_global_steps']

        real_imgs = batch['x'].type(torch.cuda.FloatTensor)
        b_size = len(real_imgs)

        noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (b_size, latent_dim)))

        optim_dis.zero_grad()
        real_valid=discriminator(real_imgs)
        fake_imgs = generator(noise).detach()
        fake_valid = discriminator(fake_imgs)

        if args.loss == 'hinge':
            loss_dis = torch.mean(nn.ReLU(inplace=True)(1.0 - real_valid)).to(device) + torch.mean(nn.ReLU(inplace=True)(1 + fake_valid)).to(device)
        elif args.loss == 'wgangp_eps':
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs.detach(), args.phi)
            loss_dis = -torch.mean(real_valid) + torch.mean(fake_valid) + gradient_penalty * 10 / (args.phi ** 2)         

        loss_dis.backward()
        optim_dis.step()

        writer.add_scalar("loss_dis", loss_dis.item(), global_steps)
        print("loss_dis", loss_dis.item(), global_steps)
        if global_steps % n_critic == 0:

            optim_gen.zero_grad()
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            gener_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (gener_batch_size, latent_dim)))

            generated_imgs= generator(gener_noise)
            fake_valid = discriminator(generated_imgs)

            gener_loss = -torch.mean(fake_valid).to(device)
            gener_loss.backward()
            optim_gen.step()
            writer.add_scalar("gener_loss", gener_loss.item(), global_steps)
            print("gener_loss", gener_loss.item(), global_steps)
            gen_step += 1

     




best = 1e4

parameters = dict()

parameters['dataset'] = 'charades'
parameters['num_frames'] = init_size**2
    
datasets = get_datasets(parameters)
dataset = datasets["train"]
batch_size = 40
train_loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=10, drop_last=True, collate_fn=collate)


for epoch in range(args.epoch):

    lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None

    train(generator, discriminator, train_loader, optim_gen, optim_dis,
    epoch, writer, lr_schedulers,img_size=32, latent_dim = args.latent_dim,
    n_critic = args.n_critic,
    gener_batch_size=args.gener_batch_size)

    checkpoint = {'epoch':epoch, 'best_fid':best}
    checkpoint['generator_state_dict'] = generator.state_dict()
    checkpoint['discriminator_state_dict'] = discriminator.state_dict()



checkpoint = {'epoch':epoch, 'best_fid':best}
checkpoint['generator_state_dict'] = generator.state_dict()
checkpoint['discriminator_state_dict'] = discriminator.state_dict()
save_checkpoint(checkpoint,is_best=False, output_dir=args.output_dir)
