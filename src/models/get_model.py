import importlib

JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]

LOSSES = ["rc", "kl", "rcxyz"]  # not used: "hp", "mmd", "vel", "velxyz"

MODELTYPES = ["cvae"]  # not used: "cae"
ARCHINAMES = ["fc", "gru", "transformer", "transformermemory", "transformermulti", "transformerrecurrent","transformerrecurrentallz", "transformerautoregressive","transformerrecurrentoneaction" ,"transformercluster", "transformersinglez","transformerrecurrentperact", "transformerrecurrentperactbs", "transformerrecurrentperactoneact", "transformerpartrecurrent", "transformerpartrecurrent1", "transgru", "grutrans", "autotrans"]

#from src.models.modeltype.TransGAN import Discriminator
import torch

def get_model(parameters):
    modeltype = parameters["modeltype"]
    archiname = parameters["archiname"]

    archi_module = importlib.import_module(f'.architectures.{archiname}', package="src.models")
    Encoder = archi_module.__getattribute__(f"Encoder_{archiname.upper()}")
    Decoder = archi_module.__getattribute__(f"Decoder_{archiname.upper()}")

    model_module = importlib.import_module(f'.modeltype.{modeltype}', package="src.models")
    Model = model_module.__getattribute__(f"{modeltype.upper()}")

    encoder = Encoder(**parameters)
    decoder = Decoder(**parameters)
    

    input_dim = 22*6
    patch_size = 64
    heads = 3
    init_size = 8

    #disc = Discriminator(diff_aug = None, image_size=32, patch_size=patch_size, input_channel=3, num_classes=1,
    #             dim=input_dim, depth=7, heads=heads, mlp_ratio=4,
    #             drop_rate=0., num_frames=patch_size-1)

    #disc  = TransformerDiscriminator(**parameters)

#    disc = Discriminator_R()
    disc = None

    #disc_vert = Discriminator(diff_aug = None, image_size=32, patch_size=patch_size, input_channel=3, num_classes=1,
    #             dim=22*3*10475, depth=4, heads=3, mlp_ratio=4,
    #             drop_rate=0., num_frames=patch_size-1)
    disc_vert = None

    #disc = Discriminator_R()
    #disc_vert = Discriminator_R_Vert()
    parameters["outputxyz"] = "rcxyz" in parameters["lambdas"]
    model = Model(encoder, decoder, **parameters, disc=disc, disc_vert = disc_vert).to(parameters["device"])
    #model = nn.DataParallel(model, device_ids=[1])#.to(device)

    if False:
        checkpointpath = 'exps/recurrent_proxrefined_unlimitted/checkpoint_0150.pth.tar'
        state_dict = torch.load(checkpointpath, map_location=parameters["device"])
        model.load_state_dict(state_dict)
    return model
