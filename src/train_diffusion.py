from pathlib import Path
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
import argparse
import wandb


def build_dataset_dataloader(repo_id: str, root: str, dataset_fps, batch_size, deltas, device
                             ) -> tuple[LeRobotDataset, torch.utils.data.DataLoader]:

    # horizon:
    # -1 -> from the past
    # 0 -> what we try to predict
    # 1 ++ -> from the future
    # with deltas=list(range(-1, 7)), we will predict 6 steps into the future and use 1 step from the past

    deltas = [delta * 1/dataset_fps for delta in deltas]
    delta_timestamps = {
        "action": deltas,
        "observation.image_primary": deltas,
        "observation.image_wrist": deltas,
        "observation.state": deltas,
    }

    dataset = LeRobotDataset(repo_id,
                             root,
                             local_files_only=True,
                             delta_timestamps=delta_timestamps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    return dataset, dataloader


def build_airnet_diffusion_policy(dataset: LeRobotDataset) -> DiffusionPolicy:
    n_deltas = len(dataset.delta_timestamps["action"])

    airnet_cfg = DiffusionConfig(
        input_shapes={
            # "action": [7],
            "observation.image_primary": (3, 480, 640),
            "observation.image_wrist": (3, 480, 640),
            "observation.state": [11]},
        output_shapes={"action": [7]},
        input_normalization_modes={"observation.image_primary": "mean_std",
                                   "observation.image_wrist": "mean_std",
                                   "observation.state": "min_max",
                                   },
        output_normalization_modes={"action": "min_max"},
        crop_shape=[440, 560],
        horizon=n_deltas,
        n_obs_steps=n_deltas,
    )

    return DiffusionPolicy(airnet_cfg, dataset_stats=dataset.meta.stats)


def train_diffusion_policy(policy: DiffusionPolicy, dataloader, device, training_steps=5000,
                           log_freq=25, save_dir=None, save_wdb=False) -> DiffusionPolicy:
    step = -1
    done = False

    policy.train()
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    while not done:
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
                if save_wdb:
                    wandb.log({"loss": loss.item(), "step": step})
            if step >= training_steps:
                done = True
                break

    if save_dir is not None:
        policy.save_pretrained(save_dir)

    return policy


def run(args):
    wandb.init(project="diffusion policy", name=args.name)

    device = torch.device("cuda")

    dataset, dataloader = build_dataset_dataloader(
        repo_id=f"airnet/{args.name}",
        root=args.data_root_dir,
        dataset_fps=args.dataset_fps,
        batch_size=args.batch_size,
        deltas=list(range(-1, args.deltas_limit)),
        device=device)

    policy = build_airnet_diffusion_policy(dataset)
    policy = train_diffusion_policy(policy, dataloader, device, save_dir=args.run_root_dir, save_wdb=True)
    return policy


if __name__ == "__main__":
    # dataset_to_use = "pick_up_blue_200"
    # output_directory = Path(f"../outputs/train_diffusion/{dataset_to_use}")
    # output_directory.mkdir(parents=True, exist_ok=True)
    # device = torch.device("cuda")

    # horizon = list(range(-1, 7))
    # dataset_fps = 15

    # dataset, dataloader = build_dataset_dataloader(
    #     repo_id=f"airnet/{dataset_to_use}",
    #     root=f"../datasets/lerobot_datasets/{dataset_to_use}",
    #     dataset_fps=dataset_fps,
    #     batch_size=16,
    #     deltas=horizon)

    # policy = build_airnet_diffusion_policy(dataset)
    # policy = train_diffusion_policy(policy, dataloader, device, save_dir=output_directory)
    parser = argparse.ArgumentParser(description='Diffusion model name, data root directory, and run root directory')
    parser.add_argument('--name', type=str)
    parser.add_argument('--data_root_dir', type=str)
    parser.add_argument('--run_root_dir', type=str)
    parser.add_argument('--dataset_fps', type=int, default=15)
    parser.add_argument('--deltas_limit', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=5000)
    # parser.add_argument('--save_steps', type=int, default=1000)
    args = parser.parse_args()

    if args.data_root_dir is None:
        args.data_root_dir = f"datasets/lerobot_datasets/{args.name}"
    if args.run_root_dir is None:
        args.run_root_dir = f"outputs/train_diffusion/{args.name}"

    print(args)
    run(args)
