import os
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from sh_utils import eval_complex_sh, scatter2ComplexSH
from torch.utils.tensorboard import SummaryWriter
import debugpy

debugpy.listen(5678)
print("Waiting for debugger attach")
debugpy.wait_for_client()

parser = argparse.ArgumentParser(description='SH optimization')
parser.add_argument('--model_save_path', type=str, default='.', help='Path to save the model')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--eval', action='store_true', help='Evaluate the model')
args = parser.parse_args()

if os.path.exists(args.model_save_path) is False:
    os.makedirs(args.model_save_path)
if os.path.exists(os.path.join(args.model_save_path, 'Checkpoints')) is False:
    os.makedirs(os.path.join(args.model_save_path, 'Checkpoints'))

tb_writer = SummaryWriter(os.path.join(args.model_save_path, 'logs'))

def get_cylinder_cameras(num_positions, radius, height):
    """Generate points on a cylinder."""
    n_rotations = int(height / (2 * torch.pi * radius) * torch.sqrt(torch.tensor(num_cameras)))
    t = torch.linspace(0, n_rotations * 2 * torch.pi, num_positions)
    
    x = radius * torch.cos(t)
    y = radius * torch.sin(t)
    z = torch.linspace(-height/2, height/2, num_positions)
    return torch.column_stack((x, y, z))


def model(deg, coefficients, dirs):
    return eval_complex_sh(deg, coefficients, dirs)

def save_sh_parameters(sh_features, active_sh_degree, filename="sh_features.pt"):
    """
    Save SH parameters as a .pt file.
    Args:
        sh_features (torch.Tensor): the learnable SH parameter tensor.
        active_sh_degree (int): the active SH degree.
        filename (str): the filename to save to, e.g. "sh_features.pt".
    """
    dict_to_save = {
        "sh_features": sh_features,
        "active_sh_degree": active_sh_degree
    }
    torch.save(dict_to_save, filename)

def complex_mse_loss(input, target):
    """
    Compute the mean squared error between two complex tensors.
    Args:
        input (torch.Tensor): the input tensor.
        target (torch.Tensor): the target tensor.
    Returns:
        torch.Tensor: the mean squared error.
    """
    mse = ((input.real - target.real)**2 + (input.imag - target.imag)**2).mean()
    return mse #* torch.tanh(mse/2)

def create_smooth_reflection(dirs):
    """
    Generate a smooth, low-frequency complex reflection function for each direction in dirs.

    Args:
        dirs (torch.Tensor): shape (N, 3), normalized directions (x,y,z).
    Returns:
        torch.complex64: shape (N,), complex reflection at each direction.
    """
    # Extract x,y,z
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]

    # Convert to spherical angles
    theta = torch.acos(z)            # in [0, π]
    phi   = torch.atan2(y, x)        # in [-π, π] (or [0, 2π) depending on usage)

    # Choose some constants:
    c0 = 0.5
    c1 = 0.3
    c2 = 0.4

    # Real part: c0 + c1*cos(2θ) + c2*cos(3φ)
    real_part = c0 + c1*torch.cos(theta) + c2*torch.cos(2.0 * phi)

    # Imag part: c1*sin(1θ) + c2*sin(3φ) (just an example)
    #   You could also do something like c1*sin(2θ) if you want consistent frequency with real part
    imag_part = c1*torch.sin(theta) + c2*torch.sin(2.0 * phi)

    reflection = torch.complex(real_part, imag_part)
    return reflection

num_points = 9
num_cameras = 1000
radius = 0.85
height = 1.5 * radius
cylinder_cameras = get_cylinder_cameras(num_cameras, radius, height).to('cuda')

train_cameras = cylinder_cameras[:num_cameras-4]
test_cameras = cylinder_cameras[num_cameras-4:]

#################### Define SH parameters for optimization ####################
max_sh_degree = 6
active_sh_degree = 2

# Define 30 points "somewhere around" the given locations
point_inside = torch.tensor([
    [0.0, 0.0, 1.0], [0.0, 0.1, 0.1], [0.1, 0.0, 0.0],
    [0.2, 0.3, 0.0], [0.09, 0.43, 0.12], [0.2, 0.1, 0.1],
    [0.134, 0.23, 0.056], [0.015, 0.0089, 0.31], [0.1, 0.1, 0.1]
    # [0.0033, 0.123, 0.232], [0.01, 0.02, 0.95], [0.05, 0.12, 0.11],
    # [0.09, 0.02, 0.02], [0.22, 0.32, 0.02], [0.085, 0.45, 0.15],
    # [0.25, 0.12, 0.09], [0.13, 0.25, 0.07], [0.02, 0.015, 0.33],
    # [0.12, 0.08, 0.15], [0.005, 0.13, 0.24], [0.03, 0.04, 0.91],
    # [0.06, 0.15, 0.12], [0.11, 0.03, 0.03], [0.23, 0.35, 0.03],
    # [0.075, 0.44, 0.18], [0.27, 0.11, 0.12], [0.14, 0.27, 0.09],
    # [0.025, 0.02, 0.36], [0.15, 0.12, 0.13], [0.007, 0.135, 0.26]
], dtype=torch.float32).to('cuda')


if args.train:

    batch_size = 32

    # # for every camera, define a complex random value ranging from -1 to 1 (numpoints, numcameras)
    # real = torch.rand((num_points, num_cameras), dtype=torch.float32) * 2 - 1
    # imag = torch.rand((num_points, num_cameras), dtype=torch.float32) * 2 - 1
    # reflection_per_direction = torch.complex(real, imag).to('cuda')

    dir_vecs = train_cameras[None, :, :] - point_inside[:, None, :]
    dir_vecs = dir_vecs / torch.norm(dir_vecs, dim=-1, keepdim=True)
    reflection_per_direction = create_smooth_reflection(dir_vecs)

    backprojected_value = reflection_per_direction.sum(dim=-1).to('cuda') #(numpoints,)

    ######################################################################
    #define learnable parameters for optimization

    scatter_SH = scatter2ComplexSH(backprojected_value/len(train_cameras))
    features = torch.zeros((scatter_SH.shape[0], (max_sh_degree + 1) ** 2), dtype=torch.complex64, device="cuda")
    features[:, 0] = scatter_SH

    sh_features = nn.Parameter(features.contiguous(),requires_grad=True)

    initial_lr = 0.01
    min_lr = 1e-6
    first_cycle_steps = 2000  # T_0: Length of first cycle
    cycle_mult = 1  # T_mult: Factor to increase cycle length after each 

    # define the optimizer
    optimizer = torch.optim.Adam([sh_features], lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=first_cycle_steps,
        T_mult=cycle_mult,
        eta_min=min_lr
    )

    # define the number of iterations
    num_iterations = 10000
    first_iter = 0
    iters = len(range(first_iter, num_iterations))
    pro_bar = tqdm(range(first_iter, num_iterations), desc="Optimizing SHs", leave=True)
    first_iter += 1
    for iteration in range(first_iter, num_iterations+1):

        #randomly select a camera 
        camera_idx = torch.randint(0, len(train_cameras), (batch_size,))
        camera = train_cameras[camera_idx]

        if iteration % 2000 == 0:
            if active_sh_degree < max_sh_degree:
                active_sh_degree += 1

        loss_ = torch.zeros((num_points, 1), dtype=torch.float32, device="cuda")

        for idx in range(num_points):

            optimizer.zero_grad()

            dir_vec = camera - point_inside[idx]
            dir_vec = dir_vec / torch.norm(dir_vec, dim=-1, keepdim=True)

            sh_feat = sh_features[idx].unsqueeze(0)

            #forward model
            forward = model(active_sh_degree, sh_feat, dir_vec)

            gt = reflection_per_direction[idx, camera_idx]

            #calculate loss
            # loss = complex_mse_loss(forward, gt)
            loss =(torch.nn.functional.l1_loss(forward.real, gt.real) + torch.nn.functional.l1_loss(forward.imag, gt.imag)).mean()

            loss_[idx] = loss

        loss_ = loss_.mean()

        #backpropagate
        loss_.backward()
        optimizer.step()

        if iteration % first_cycle_steps == 0:
            scheduler.base_lrs[0] = scheduler.base_lrs[0] * 0.5

        scheduler.step(iteration + idx / iters)

        current_lr = optimizer.param_groups[0]['lr']
        pro_bar.set_postfix({
            "Loss": loss.item(),
            "LR": current_lr,
            "SH Degree": active_sh_degree
        })
        pro_bar.update()


        tb_writer.add_scalar("Loss", loss_.item(), iteration)
        tb_writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], iteration)


        if iteration % 100 == 0:
            save_sh_parameters(sh_features, active_sh_degree, filename=os.path.join(args.model_save_path, 'Checkpoints', f"sh_features_{iteration}.pt"))

    # torch.save(reflection_per_direction, os.path.join(args.model_save_path, 'reflection_per_direction.pt'))

    tb_writer.close()
    pro_bar.close()

if args.eval:
    # reflection_per_direction = torch.load(os.path.join(args.model_save_path, 'reflection_per_direction.pt'))
    state_dict = torch.load(os.path.join(args.model_save_path, 'Checkpoints', 'sh_features_10000.pt'))

    # reflection_per_direction = create_smooth_reflection(cylinder_cameras[None, :, :] - point_inside[:, None, :])

    sh_features = state_dict["sh_features"]
    sh_degree = state_dict["active_sh_degree"]

    num_points = point_inside.shape[0]
    num_cameras = test_cameras.shape[0]

    total_mae = 0.0
    total_rmse = 0.0
    total_sum_error = 0.0

    for idx in range(num_points):
        mae = 0.0
        mse = 0.0
        predicted_sums = []
        ground_truth_sums = []

        for camera_idx in tqdm(range(len(test_cameras))):
            camera = test_cameras[camera_idx]

            dir_vec = camera - point_inside[idx]
            dir_vec = dir_vec / torch.norm(dir_vec, dim=-1, keepdim=True)

            reflection_per_direction = create_smooth_reflection(dir_vec)

            sh_feat = sh_features[idx].unsqueeze(0)

            # Forward model
            forward = model(sh_degree, sh_feat, dir_vec)

            # Ground truth
            gt = reflection_per_direction

            print("Forward: ", forward.item(), "GT: ", gt.item())

            # Compute individual errors
            mae += (forward.real - gt.real).abs().mean().item()
            mae += (forward.imag - gt.imag).abs().mean().item()

            mse += ((forward.real - gt.real) ** 2).mean().item()
            mse += ((forward.imag - gt.imag) ** 2).mean().item()

            predicted_sums.append(forward)
            ground_truth_sums.append(gt)

        # Summing over all directions for a single point
        forward_sum = torch.stack(predicted_sums).sum(dim=0)
        gt_sum = torch.stack(ground_truth_sums).sum(dim=0)

        print("Forward Sum: ", forward_sum, "GT Sum: ", gt_sum)

        # Calculate sum error
        sum_error = (forward_sum.real - gt_sum.real).abs().mean().item() + \
                    (forward_sum.imag - gt_sum.imag).abs().mean().item()

        total_sum_error += sum_error

        # Aggregate metrics
        total_mae += mae / num_cameras
        total_rmse += torch.sqrt(torch.tensor(mse / num_cameras)).item()

        print(f"Point {idx + 1}/{num_points}: MAE={mae/num_cameras:.6f}, RMSE={torch.sqrt(torch.tensor(mse / num_cameras)):.6f}, "
              f"Sum Error={sum_error:.6f}")

    # Average metrics across all points
    avg_mae = total_mae / num_points
    avg_rmse = total_rmse / num_points
    avg_sum_error = total_sum_error / num_points

    print("\n--- Evaluation Summary ---")
    print(f"Mean Absolute Error (MAE): {avg_mae:.6f}")
    print(f"Root Mean Square Error (RMSE): {avg_rmse:.6f}")
    print(f"Average Total Sum Error: {avg_sum_error:.6f}")
