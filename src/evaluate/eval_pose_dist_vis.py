#evaluates the pose in the space of the SMPL parameters

import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from src.utils.get_model_and_data import get_model_and_data
from src.models.get_model import get_model
from src.models.smpl import SMPL,SMPLX, JOINTSTYPE_ROOT
from src.render.renderer import get_renderer
import src.evaluate.eval_functions as ef


from src.parser.generate import parser
from src.utils.fixseed import fixseed # noqa

def generate_actions(beta, model, dataset, epoch, params, folder, data_ind, renderer, num_frames=60,
                     durationexp=False, vertstrans=True, onlygen=False, nspa=10, inter=False, writer=None):
    """ Generate & viz samples """

    onlygen = True

    # visualize with joints3D
    model.outputxyz = True
    # print("remove smpl")
    model.param2xyz["jointstype"] = "vertices"

    print("Visualization of the epoch {epoch}")

    fact = params["fact_latent"]
    num_classes = dataset.num_classes + 50
    #todo 
    num_classes = 1
    classes = torch.arange(num_classes)
    #classes = torch.from_numpy(np.asarray([0,0,0,0,0,0,0]))
    if not onlygen:
        nspa = 1

    nats = num_classes
    #num_frames = 400
    durationexp = False
    if durationexp:
        nspa = 4
        durations = [40, 60, 80, 100]
        gendurations = torch.tensor([[dur for cl in classes] for dur in durations], dtype=int)
    else:
        gendurations = torch.tensor([num_frames for cl in classes], dtype=int)
    
    real_samples, mask_real, real_lengths, labels, act_time_stamps, frame_act_map = dataset.get_label_sample_ind(data_ind)
        # to visualize directly
    classes = labels
    if True:
        # extract the real samples
        
        print('classes', classes)
        # Visualizaion of real samples
        visualization = {"x": real_samples.to(model.device),
                         "y": classes.to(model.device),
                         "mask": mask_real.to(model.device),
                         "lengths": real_lengths.to(model.device),
                         "output": real_samples.to(model.device),
                         "action_timestamps": act_time_stamps,
                         "frame_act_map": frame_act_map}

        reconstruction = {"x": real_samples.to(model.device),
                          "y": classes.to(model.device),
                          "lengths": real_lengths.to(model.device),
                          "mask": mask_real.to(model.device),
                          "action_timestamps": act_time_stamps,
                          "frame_act_map": frame_act_map}

    print("Computing the samples poses..")

    # generate the repr (joints3D/pose etc)
    model.eval()
    with torch.no_grad():
        if inter:
            noise_same_action = "interpolate"
        else:
            noise_same_action = "random"

        noise_diff_action = "random"

        # Generate the new data
        generation = model.generate(classes, gendurations, nspa=nspa,
                                    noise_same_action=noise_same_action,
                                    noise_diff_action=noise_diff_action,
                                    fact=fact, action_timestamps=act_time_stamps, frame_act_map= frame_act_map)

        generation["output_xyz"] = model.rot2xyz(generation["output"],
                                                     generation["mask"], vertstrans=vertstrans,
                                                     beta=beta)
        output = generation["output_xyz"].reshape(nspa, nats, *generation["output_xyz"].shape[1:]).cpu().numpy()

    # if True: #todo
    #      for s in range(output.shape[0]):
    #             current_fn = os.path.join('matching', )
    #         vid_out = cv2.VideoWriter(current_fn, apiPreference=0,fourcc=0x7634706d, fps=30,frameSize = (width, height))
    #         all_verts =  out[s].transpose([2,0,1])
    #         for l in range(len(all_verts)):
    #             image = renderer.render(background, all_verts[l], cam, color=color)
    #             vid_out.write(image)
    #         vid_out.release()


 
        
    return generation, real_samples, labels.to(real_samples.device), mask_real.to(real_samples.device), output


def main():
    parameters, folder, checkpointname, epoch = parser()
    nspa = parameters["num_samples_per_action"]

    # no dataset needed
    if parameters["mode"] in []:   # ["gen", "duration", "interpolate"]:
        model = get_model(parameters)
    else:
        model, datasets = get_model_and_data(parameters)
        dataset = datasets["train"]  # same for ntu

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    model.load_state_dict(state_dict)

    #from src.utils.fixseed import fixseed  # noqa
    all_real_samples = []
    all_gen_samples = []
    all_num_frames = []
    all_labels = []
    all_rendered_output = []
    max_num_frames = dataset.max_len

    max_num_classes = dataset.max_num_classes
    #for seed in [1 ,2,3,4]: #5,67,8,9,10, 13, 14, 15, 16, 17, 18, 19, 20,21,22]:  # [0, 1, 2]:
    #    fixseed(seed)
    fixseed(1071)
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
        
    for m in range(len(dataset._train)):
        if True and m == 2:
                break
        data_ind = np.random.randint(len(dataset._train)-1)

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
            generation, real_samples, labels, real_mask, rendered_output = generate_actions(beta, model, dataset, epoch, parameters,
                                      folder, data_ind, renderer, inter=inter, vertstrans=vertstrans,
                                      nspa=nspa, onlygen=onlygen)
            all_num_frames.append(real_mask.sum())
            gen_output = (generation['output'].to(real_samples.device).permute(0,3,1,2))
            real_mask = real_mask.unsqueeze(2).unsqueeze(3).expand(gen_output.shape)

            real_samples = real_mask*(real_samples.permute(0,3,1,2))

            to_pad = max_num_frames - real_samples.shape[1]
            if to_pad > 0:
                padding_mat = torch.zeros(real_samples.shape[0], to_pad, real_samples.shape[2], real_samples.shape[3]).to(real_samples.device)
                real_samples = torch.cat([real_samples, padding_mat], dim=1)
                gen_output = torch.cat([gen_output, padding_mat], dim=1)
            
            to_pad = max_num_classes - labels.shape[1]
            if to_pad > 0:
                pad_vec = torch.zeros(len(labels), to_pad)-1
                labels = torch.cat([labels, pad_vec], dim=1)
            all_labels.append(labels)
            
            all_gen_samples.append(gen_output)
            all_real_samples.append(real_samples)
            all_rendered_output.append(rendered_output)


    all_num_frames = np.asarray(all_num_frames)
    all_real_samples = torch.stack(all_real_samples)[:,0]
    all_gen_samples = torch.stack(all_gen_samples)[:,0]
    all_labels = torch.cat(all_labels)

    print('number of valid classes', (all_labels!=-1).sum())
    #min_smp_dist_param, total_match, num_matched_labels = ef.calc_min_distance_param(all_real_samples, all_gen_samples, all_num_frames, all_labels.cpu().numpy())

    assignment, min_smp_dist_param_opt, total_match_opt, num_matched_labels_opt = ef.calc_min_distance_param_assignment(all_real_samples, all_gen_samples, all_num_frames, all_labels.cpu().numpy())

    dist_to_gt_param = ef.calc_distance_to_gt_param(all_real_samples, all_gen_samples, all_num_frames)

    all_pair_dist_param = ef.calc_distance_all_pairs_param(all_real_samples, all_gen_samples, all_num_frames)
    #print('min dist:', min_smp_dist_param.item(), ', num matches:', total_match, ', num matched labels:', num_matched_labels.item(), ', min dist optimal:', min_smp_dist_param_opt.item(), ', num matches opt:', total_match_opt.item(), 'num matched labels opt:', num_matched_labels_opt.item(), 'dist to gt:', dist_to_gt_param.item(), 'average all pairs dist:', all_pair_dist_param.item())
    print('min dist optimal:', min_smp_dist_param_opt.item(), ', num matches opt:', total_match_opt.item(), 'num matched labels opt:', num_matched_labels_opt.item(), 'dist to gt:', dist_to_gt_param.item(), 'average all pairs dist:', all_pair_dist_param.item())
   #tensor(1.9641) 52 tensor(17.3888) tensor(53) tensor(18.1650) tensor(31.8613)

    a = 0



if __name__ == '__main__':
    main()
