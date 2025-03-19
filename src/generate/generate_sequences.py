import os

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from src.utils.get_model_and_data import get_model_and_data
from src.models.get_model import get_model
from src.models.smpl import SMPL,SMPLX, JOINTSTYPE_ROOT
from src.render.renderer import get_renderer


from src.parser.generate import parser
from src.utils.fixseed import fixseed # noqa

plt.switch_backend('agg')

def generate_actions(beta, model, dataset, epoch, params, folder, num_frames=60,
                     durationexp=False, vertstrans=True, onlygen=False, nspa=10, inter=False, writer=None):
    """ Generate & viz samples """

    onlygen = True

    # visualize with joints3D
    model.outputxyz = True
    # print("remove smpl")
    model.param2xyz["jointstype"] = "vertices"

    print(f"Visualization of the epoch {epoch}")

    fact = params["fact_latent"]
    num_classes = dataset.num_classes + 10  
    #todo 
    #num_classes = 1
    classes = torch.arange(num_classes)
    #classes = torch.from_numpy(np.asarray([0,0,0,0,0,0,0]))
    if not onlygen:
        nspa = 1

    nats = num_classes
    num_frames = 60
    durationexp = False
    if durationexp:
        nspa = 4
        durations = [40, 60, 80, 100]
        gendurations = torch.tensor([[dur for cl in classes] for dur in durations], dtype=int)
    else:
        gendurations = torch.tensor([num_frames for cl in classes], dtype=int)

    real_samples, mask_real, real_lengths, labels = dataset.get_label_sample_batch(classes.numpy())
    act_time_stamps = None
        # to visualize directly
    classes = labels
    if not onlygen:
        # extract the real samples
        
        print('classes', classes)
        # Visualizaion of real samples
        visualization = {"x": real_samples.to(model.device),
                         "y": classes.to(model.device),
                         "mask": mask_real.to(model.device),
                         "lengths": real_lengths.to(model.device),
                         "output": real_samples.to(model.device),
                         "action_timestamps": act_time_stamps}

        reconstruction = {"x": real_samples.to(model.device),
                          "y": classes.to(model.device),
                          "lengths": real_lengths.to(model.device),
                          "mask": mask_real.to(model.device),
                          "action_timestamps": act_time_stamps}

    print("Computing the samples poses..")

    # generate the repr (joints3D/pose etc)
    model.eval()
    with torch.no_grad():
        if not onlygen:
            # Get xyz for the real ones
            visualization["output_xyz"] = model.rot2xyz(visualization["output"],
                                                        visualization["mask"],
                                                        vertstrans=vertstrans,
                                                        beta=beta)

            # Reconstruction of the real data
            reconstruction = model(reconstruction)  # update reconstruction dicts

            noise_same_action = "random"
            noise_diff_action = "random"

            # Generate the new data
        
            generation = model.generate(classes, gendurations, nspa=nspa,
                                        noise_same_action=noise_same_action,
                                        noise_diff_action=noise_diff_action,
                                        fact=fact)

            generation["output_xyz"] = model.rot2xyz(generation["output"],
                                                     generation["mask"], vertstrans=vertstrans,
                                                     beta=beta)

            outxyz = model.rot2xyz(reconstruction["output"],
                                   reconstruction["mask"], vertstrans=vertstrans,
                                   beta=beta)
            reconstruction["output_xyz"] = outxyz
        else:
            if inter:
                noise_same_action = "interpolate"
            else:
                noise_same_action = "random"

            noise_diff_action = "random"

            # Generate the new data
            generation = model.generate(classes, gendurations, nspa=nspa,
                                        noise_same_action=noise_same_action,
                                        noise_diff_action=noise_diff_action,
                                        fact=fact)

            generation["output_xyz"] = model.rot2xyz(generation["output"],
                                                     generation["mask"], vertstrans=vertstrans,
                                                     beta=beta)
            output = generation["output_xyz"].reshape(nspa, nats, *generation["output_xyz"].shape[1:]).cpu().numpy()

    if not onlygen: #todo
        output = np.stack([visualization["output_xyz"].cpu().numpy(),
                           generation["output_xyz"][:,:,:,0:60].cpu().numpy(),
                           reconstruction["output_xyz"].cpu().numpy()])

    width = 1024
    height = 1024
    cam=(0.75, 0.75, 0, 0.10)
    color=[0.11, 0.53, 0.8]
  
    if 'prox' in dataset.dataname: #todo
        smpl_model = SMPLX().eval().to(model.device)#TODO SMPL
        renderer = get_renderer(width, height, faces=smpl_model.body_model.faces)

    else:
        smpl_model = SMPL().eval().to(model.device)#TODO SMPL
        renderer = get_renderer(width, height, faces=None)

    sample_types = {0:'vis', 1:'gen', 2:'recon'}

  
    background = np.zeros((height, width, 3))
    save_path = os.path.join(folder, 'generation')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for id, out in enumerate(output):
        #if id == 0:
        #    continue
        #out_type = sample_types[id]
        if onlygen:
            out_type = 'gen'
        else:
            out_type = sample_types[id]
        print('generating for set %d sample_type %s'%(id, out_type))
        
        for s in range(out.shape[0]):
            current_fn = os.path.join(save_path, 'action%d%s_%s%d.mp4'%(s,dataset.label_to_action_name(classes[s]).replace('\n','')[0:100], out_type, id))
            vid_out = cv2.VideoWriter(current_fn, apiPreference=0,fourcc=0x7634706d, fps=30,frameSize = (width, height))
            all_verts =  out[s].transpose([2,0,1])
            for l in range(len(all_verts)):
                image = renderer.render(background, all_verts[l], cam, color=color)
                vid_out.write(image)
            vid_out.release()
        
    return output


def main():
    parameters, folder, checkpointname, epoch = parser()
    nspa = parameters["num_samples_per_action"]

    # no dataset needed
    if parameters["mode"] in []:   # ["gen", "duration", "interpolate"]:
        model = get_model(parameters)
    else:
        model, datasets = get_model_and_data(parameters)
        dataset = datasets["train"]  # same for ntu

    #dataset = datasets["train"] 
    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    model.load_state_dict(state_dict)

    from src.utils.fixseed import fixseed  # noqa
    for seed in [1]:  # [0, 1, 2]:
        fixseed(seed)
        # visualize_params
        onlygen = True
        vertstrans = False
        inter = True and onlygen
        varying_beta = False
        if varying_beta:
            betas = [-2, -1, 0, 1, 2]
        else:
            betas = [0]
        for beta in betas:
            output = generate_actions(beta, model, dataset, epoch, parameters,
                                      folder, inter=inter, vertstrans=vertstrans,
                                      nspa=nspa, onlygen=onlygen)
            if varying_beta:
                filename = "generation_beta_{}.npy".format(beta)
            else:
                filename = "generation.npy"

            filename = os.path.join(folder, filename)
            np.save(filename, output)
            print("Saved at: " + filename)


if __name__ == '__main__':
    main()
