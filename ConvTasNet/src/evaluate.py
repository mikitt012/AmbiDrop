#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os

import librosa
import numpy as np
import torch

from data import SimDS_preprocessed, MatDatasetTest, MatDataset, SimDS, MatDatasetTest_ASM
from pit_criterion import cal_loss
from conv_tasnet import ConvTasNet
import conv_tasnet_ic
from utils import remove_pad
from pesq import pesq
from pystoi import stoi
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.io import loadmat, savemat

import wandb
wandb.login()

dropout = "SHChannelDropout"

if dropout == "SHChannelDropout":
    # checkpoint_path = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/ConvTasNet/checkpoints/run_2026-04-07_10-18" # for uniform dropout
    checkpoint_path = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/ConvTasNet/checkpoints/run_2026-04-09_08-35" # same, more epochs
else:
    checkpoint_path = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/ConvTasNet/checkpoints/run_2026-04-07_15-27" # for pr ch dropout

full_checkpoint_path = os.path.join(checkpoint_path, "final.pth.tar")

parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
parser.add_argument('--model_path', type=str, default=full_checkpoint_path,
                    help='Path to model file created by training')
# parser.add_argument('--data_dir', type=str, required=True,
#                     help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--cal_sdr', type=int, default=0,
                    help='Whether calculate SDR, add this option because calculation of SDR is very slow')
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--sample_rate', default=16000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')

def pick_reference_id(test_type):
    if test_type == "front hemisphere1 (rigid) radius = 0.1":
        ref_idx = 1

    if test_type == "full circle (rigid) radius = 0.1":
        ref_idx = 1

    if test_type == "planar":
        ref_idx = 6

    if test_type == "random 2D array1 radius = 0.1":
        ref_idx = 6

    if test_type == "random sphere1 radius = 0.1":
        ref_idx = 7

    if test_type == "random sphere3 (rigid) radius = 0.1":
        ref_idx = 4

    if test_type == "random sphere5 (rigid) radius = 0.05":
        ref_idx = 2

    if test_type == "semi circle planar radius = 0.05":
        ref_idx = 6

    if test_type == "ULA along X-axis":
        ref_idx = 7

    if test_type == "uniform sphere (rigid) radius = 0.1":
        ref_idx = 2

    if test_type == "front hemisphere2 (rigid) radius = 0.1":
        ref_idx = 1

    if test_type == "planar (rot=45deg)":
        ref_idx = 5

    if test_type == "random 2D array2 radius = 0.1":
        ref_idx = 2

    if test_type == "random sphere2 radius = 0.1":
        ref_idx = 2

    if test_type == "random sphere4 (rigid) radius = 0.1":
        ref_idx = 7

    if test_type == "random sphere6 (rigid) radius = 0.05":
        ref_idx = 4

    if test_type == "semi circle (rigid) radius = 0.1":
        ref_idx = 4

    if test_type == "ULA along Z-axis":
        ref_idx = 4

    if test_type == "uniform sphere (rigid) radius = 0.05":
        ref_idx = 2

    if test_type == "semi circle planar radius = 0.1":
        ref_idx = 6

    if test_type == "Aria on rigid sphere (simulated)":
        ref_idx = 3

    if test_type == "ULA along Y-axis (tilt=30deg)":
        ref_idx = 4
        
    if test_type == "ULA along x-axis (rot=30deg)":
        ref_idx = 7

    if test_type == "ULA along y-axis":
        ref_idx = 4

    if test_type == "ULA along X-axis (tilt=20)":
        ref_idx = 7

    ref_ids = ref_idx - 1
    return ref_ids

def evaluate(args):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0

    # Load model
    # model = ConvTasNet.load_model(args.model_path)
    model = conv_tasnet_ic.TasNet.load_model(args.model_path, dropout)
    print(model)
    model.eval()
    if args.use_cuda:
        model.cuda()

    wandb_active = True

    for j in range(1,3):
        if j == 1:
            data_dir = '/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/test_of_train_ds'
        else:
            data_dir = '/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/test_of_test_ds'

        for test_idx, test_type in enumerate(sorted(os.listdir(data_dir))):
        # for t in range(0,1):
            # if test_type == "ULA along X-axis":
            #     continue
            # if test_type == "Aria on rigid sphere (simulated)":
            #     continue    

            # test_type = "semi circle planar radius = 0.1"
            if test_type.startswith('.'):
                continue
            array_name = test_type.removesuffix("_preprocessed")
            name = array_name
            if wandb_active:
                wandb.init(project="ConvTasNet_experiment", entity="tatarjit-ben-gurion-university-of-the-negev",name=name)

            steering_dir = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/steering"
            mat_filename = f"{array_name}.mat"
            steer_path = os.path.join(steering_dir, mat_filename)
            steer_mat = loadmat(steer_path)
            V = steer_mat["V"]          # numpy array, shape (CH, F, Q)

            # --- 2. Load grid (theta, phi): 1 x Q ---
            grid_path = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/utils/Lebvedev2702.mat"
            grid_mat = loadmat(grid_path)
            th = grid_mat["th"].squeeze()    # shape (Q,)
            ph = grid_mat["ph"].squeeze()    # shape (Q,)

            data_path = os.path.join(data_dir, test_type)
            test_ds = MatDatasetTest_ASM(data_path, V, th, ph)
            # test_ds = MatDatasetTest(data_path)
            # test_ds = MatDataset('/gpfs0/bgu-br/projects/sim_dataset_ambisonics/si_et_05')
            # num_examples = 1
            # indices = list(range(num_examples))
            # test_ds_subset = Subset(test_ds, indices)
            # test_ds = test_ds_subset
            data_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
            
            stoi_noisy = np.array([])
            pesq_noisy = np.array([])
            stoi_enhanced = np.array([])
            pesq_enhanced = np.array([])
            sisdr_noisy = np.array([])
            sisdr_enhanced = np.array([])

            total_SISNRi = 0
            total_SISNRb = 0
            total_SISNRa = 0
            total_cnt = 0

            ref_ids = pick_reference_id(test_type)

            with torch.no_grad():
                for i, (data) in enumerate(data_loader):
                    noisy_mic, clean_mic, noisy_batch, clean_batch = data

                    # --- SILENCE CHECK ---
                    # Calculate the RMS (Root Mean Square) energy of the clean target
                    # If the audio is quieter than -60dB (approx 0.001 amplitude), skip it
                    clean_energy = torch.sqrt(torch.mean(clean_batch**2, dim=-1)) # [B]
                    if (clean_energy < 1e-4).any():
                        if i % self.print_freq == 0:
                            print(f"Skipping Batch {i}: Silent or extremely quiet clean reference detected.")
                        continue
                    
                    batch_size = noisy_batch.shape[0]  # B
                    num_samples = noisy_batch.shape[2] # T
                    mixture_lengths = torch.full((batch_size,), num_samples, dtype=torch.int64).to(noisy_batch.device)
                    
                    if args.use_cuda:
                        padded_mixture = noisy_batch.cuda() # B x C x T
                        mixture_lengths = mixture_lengths.cuda() # B
                        padded_source = clean_batch.cuda() # B x T
                        padded_source = padded_source.unsqueeze(1)
                    
                    estimate_source = model(padded_mixture) # B x 1 x T
                    
                    # loss = -si_snr(estimate_source, padded_source, debug=False)
                        
                    M,_,T = padded_mixture.shape    
                    mixture_ref = torch.chunk(padded_mixture, 9, dim =1)[0] #[M, ch, T] -> [M, 1, T]
                    mixture_ref = mixture_ref.view(M,T) #[M, 1, T] -> [M, T]
                    
                    mixture = remove_pad(mixture_ref, mixture_lengths)
                    source = remove_pad(padded_source, mixture_lengths)
                    estimate_source = remove_pad(estimate_source, mixture_lengths)
                    
                    # for each utterance
                    for mix, src_ref, src_est in zip(mixture, source, estimate_source):
                        # print("Utt", total_cnt + 1)
                        mix = np.squeeze(mix); src_ref = np.squeeze(src_ref); src_est = np.squeeze(src_est)
                        mix = mix.real.astype('float32')
                        mix = mix/mix.max(); src_ref = src_ref/src_ref.max(); src_est = src_est/src_est.max()
                        SISNRi, SISNR_before, SISNR_after = cal_SISNRi(src_ref, src_est, mix)
                        # SISNRi, SISNR_before, SISNR_after = cal_SISNRi(src_est, src_ref, mix)
                        # print("\tSI-SNRi={0:.2f}".format(SISNRi))
                        # print("\tNoisy SI-SNR={0:.2f}".format(SISNR_before))
                        total_SISNRi += SISNRi
                        total_SISNRb += SISNR_before
                        total_SISNRa += SISNR_after
                        # sisnri_array.append(SISNRi)
                        # sisnrb_array.append(SISNR_before)

                        stoi_noisy = np.append(stoi_noisy, stoi(src_ref, mix, 16000, extended=False))
                        pesq_noisy = np.append(pesq_noisy, pesq(16000, src_ref, mix, mode="wb"))
                        sisdr_noisy = np.append(sisdr_noisy, SISNR_before)

                        stoi_enhanced = np.append(stoi_enhanced, stoi(src_ref, src_est, 16000, extended=False))
                        pesq_enhanced = np.append(pesq_enhanced, pesq(16000, src_ref, src_est, mode="wb"))
                        sisdr_enhanced = np.append(sisdr_enhanced, SISNR_after)

                        total_cnt += 1

                        # # Store in the matrix instead of appending to a flat list
                        # if j == 1:
                        #     master_si_sdr_noisy[test_idx, i] = SISNR_before
                        #     master_si_sdr_enhanced[test_idx, i] = SISNR_after
                        # else:
                        #     master_si_sdr_noisy[test_idx + 10, i] = SISNR_before
                        #     master_si_sdr_enhanced[test_idx + 10, i] = SISNR_after
                if wandb_active:
                    wandb.log({
                        "test/stoi_noisy": float(stoi_noisy.mean()),
                        "test/pesq_noisy": float(pesq_noisy.mean()),
                        "test/si_sdr_noisy": float(sisdr_noisy.mean()),
                        "test/stoi_enhanced": float(stoi_enhanced.mean()),
                        "test/pesq_enhanced": float(pesq_enhanced.mean()),
                        "test/si_sdr_enhanced": float(sisdr_enhanced.mean())
                    })

                    wandb.log({
                        "audio/clean": wandb.Audio(src_ref, sample_rate=16000),
                        "audio/enhanced": wandb.Audio(src_est, sample_rate=16000),
                        "audio/noisy": wandb.Audio(mix, sample_rate=16000),
                    })
                    wandb.finish()
                
            # np.save('FaSNet/sisnri.npy',np.array(sisnri_array))
            # np.save('FaSNet/sisnrb.npy',np.array(sisnrb_array))

            print(f"---------- results for: {array_name} ----------")

            # Format: metric name  noisy -> enhanced (improvement)
            print(f"SI-SDR: {sisdr_noisy.mean():.2f} -> {sisdr_enhanced.mean():.2f} ({(total_SISNRi / total_cnt):+.2f})")
            print(f"PESQ: {pesq_noisy.mean():.2f} -> {pesq_enhanced.mean():.2f} ({pesq_enhanced.mean() - pesq_noisy.mean():+.2f})")
            print(f"STOI: {stoi_noisy.mean():.3f} -> {stoi_enhanced.mean():.3f} ({stoi_enhanced.mean() - stoi_noisy.mean():+.3f})")
            # break


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(np.squeeze(src_ref), np.squeeze(src_est))
    sisnr1b = cal_SISNR(np.squeeze(src_ref), mix)

    # src_ref = torch.from_numpy(np.squeeze(src_ref))
    # src_est = torch.from_numpy(np.squeeze(src_est))
    # mix = torch.from_numpy(np.squeeze(mix))
    # sisnr1 = si_snr(src_est.unsqueeze(0), src_ref.unsqueeze(0))
    # sisnr1b = si_snr(mix.unsqueeze(0), src_ref.unsqueeze(0))

    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    SISNRi = sisnr1 - sisnr1b
    return SISNRi, sisnr1b, sisnr1

def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    # sisnr = 10 * np.log(ratio)
    return sisnr

def si_snr(estimate: torch.Tensor, reference: torch.Tensor, epsilon=1e-8, debug=False):
    """
    Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR) between estimate and reference signals.
    
    Args:
        estimate (torch.Tensor): Estimated signal, shape [B, T]
        reference (torch.Tensor): Ground truth signal, shape [B, T]
        epsilon (float): Small value to avoid division by zero
        debug (bool): If True, print internal debugging info

    Returns:
        si_snr (torch.Tensor): SI-SNR per sample, shape [B]
    """
    if debug:
        print(f"[DEBUG] estimate shape: {estimate.shape}, reference shape: {reference.shape}")

    # 1. Zero-mean normalization (along time dimension)
    estimate = estimate - estimate.mean(dim=1, keepdim=True)
    reference = reference - reference.mean(dim=1, keepdim=True)

    if debug:
        print(f"[DEBUG] After zero-mean -> estimate: {estimate.shape}, reference: {reference.shape}")

    # 2. Compute the scaling factor
    dot = (estimate * reference).sum(dim=1, keepdim=True)  # [B, 1]
    ref_energy = (reference ** 2).sum(dim=1, keepdim=True) + epsilon  # [B, 1]

    scale = dot / ref_energy  # [B, 1]
    projection = scale * reference  # [B, T]

    # 3. Compute the noise (error)
    noise = estimate - projection  # [B, T]

    # 4. Power of target and noise
    target_power = (projection ** 2).sum(dim=1)  # [B]
    noise_power = (noise ** 2).sum(dim=1) + epsilon  # [B]

    si_snr_value = 10 * torch.log10(target_power / noise_power)  # [B]

    if debug:
        print(f"[DEBUG] target_power: {target_power.mean().item():.4f}, noise_power: {noise_power.mean().item():.4f}")
        print(f"[DEBUG] SI-SNR mean: {si_snr_value.mean().item():.4f} dB")

    return si_snr_value.mean()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    evaluate(args)


#runai-bgu submit python -n sh-convtasnet-test2 -c 20 -m 80G -g 1 --conda venv -- "python /Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/ConvTasNet/src/evaluate.py"
