import argparse
import os
import random
import numpy as np
import PIL
import torch
import torchvision
import tqdm
import datetime
from torch.utils import tensorboard
import torch_fidelity

# Import original wrapper
from torch_fidelity import GenerativeModelModuleWrapper


# ============================================================================
# 温度缩放 wrapper（支持 DaoSheng 机制）
# ============================================================================
class TemperatureWrapper(GenerativeModelModuleWrapper):
    def __init__(self, module, z_size, z_type, num_classes, temperature):
        super().__init__(module, z_size, z_type, num_classes)
        self.temperature = temperature

    def forward(self, z):
        return self.module(z, temperature=self.temperature)


# ============================================================================
# 随机种子设置（确保可复现性）
# ============================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# 生成器模型
# ============================================================================
class Generator(torch.nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.z_size = z_size
        self.conv_layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(z_size, 512, 4, stride=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),
        )

    def forward(self, z, temperature=1.0):
        logits = self.conv_layers(z.view(-1, self.z_size, 1, 1))
        output = torch.tanh(logits / temperature)
        if not self.training:
            output = 255 * (output.clamp(-1, 1) * 0.5 + 0.5)
            output = output.to(torch.uint8)
        return output


# ============================================================================
# 判别器模型（可选谱归一化）
# ============================================================================
class Discriminator(torch.nn.Module):
    def __init__(self, sn=True):
        super(Discriminator, self).__init__()
        sn_fn = torch.nn.utils.spectral_norm if sn else lambda x: x
        self.conv1 = sn_fn(torch.nn.Conv2d(3, 64, 3, stride=1, padding=(1, 1)))
        self.conv2 = sn_fn(torch.nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1)))
        self.conv3 = sn_fn(torch.nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1)))
        self.conv4 = sn_fn(torch.nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)))
        self.conv5 = sn_fn(torch.nn.Conv2d(128, 256, 3, stride=1, padding=(1, 1)))
        self.conv6 = sn_fn(torch.nn.Conv2d(256, 256, 4, stride=2, padding=(1, 1)))
        self.conv7 = sn_fn(torch.nn.Conv2d(256, 512, 3, stride=1, padding=(1, 1)))
        self.fc = sn_fn(torch.nn.Linear(4 * 4 * 512, 1))
        self.act = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        m = self.act(self.conv1(x))
        m = self.act(self.conv2(m))
        m = self.act(self.conv3(m))
        m = self.act(self.conv4(m))
        m = self.act(self.conv5(m))
        m = self.act(self.conv6(m))
        m = self.act(self.conv7(m))
        return self.fc(m.view(-1, 4 * 4 * 512))


# ============================================================================
# Hinge Loss
# ============================================================================
def hinge_loss_dis(fake, real):
    return torch.nn.functional.relu(1.0 - real).mean() + torch.nn.functional.relu(1.0 + fake).mean()

def hinge_loss_gen(fake):
    return -fake.mean()


# ============================================================================
# 阴阳平衡机制 - 仅计算缩放因子，不直接修改学习率
# ============================================================================
def compute_true_yinyang_lr_scales(D_real_outputs, D_fake_outputs, device="cuda"):
    """
    基于判别器对真实和生成样本的输出，计算 G 和 D 的自适应学习率缩放因子。
    """
    with torch.no_grad():
        mu_real = D_real_outputs.mean().item()
        mu_fake = D_fake_outputs.mean().item()

    lr_G_mult = 1.0
    lr_D_mult = 1.0

    if mu_fake < -1.0:
        lr_G_mult = 1.0 + max(0.0, -1.0 - mu_fake)
        lr_G_mult = min(lr_G_mult, 2.0)
    elif mu_fake > 0.0:
        lr_D_mult = 1.0 + 0.5 * mu_fake
        lr_D_mult = min(lr_D_mult, 1.5)

    return lr_G_mult, lr_D_mult


# ============================================================================
# 学习率管理器
# ============================================================================
class LearningRateManager:
    def __init__(self, optimizer, scheduler, base_lr, mode, enable_yinyang):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.base_lr = base_lr
        self.mode = mode
        self.enable_yinyang = enable_yinyang and mode in {"yinyang_gradscale", "yinyang_daosheng"}
        
    def get_scheduled_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def apply_yinyang_scale(self, lr_mult):
        if not self.enable_yinyang:
            return self.get_scheduled_lr()
        scheduled_lr = self.get_scheduled_lr()
        actual_lr = scheduled_lr * lr_mult
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = actual_lr
        return actual_lr
    
    def restore_scheduled_lr(self):
        return self.get_scheduled_lr()
    
    def step_scheduler(self):
        self.scheduler.step()
        
    def log_lr(self, tb_writer, step, prefix="G"):
        if tb_writer is not None:
            scheduled_lr = self.get_scheduled_lr()
            tb_writer.add_scalar(f"LR/{prefix}_scheduled", scheduled_lr, global_step=step)


# ============================================================================
# 主训练函数
# ============================================================================
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.dir_dataset, exist_ok=True)
    ds_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ds_instance = torchvision.datasets.CIFAR10(
        args.dir_dataset, train=True, download=True, transform=ds_transform
    )
    loader = torch.utils.data.DataLoader(
        ds_instance, batch_size=args.batch_size, drop_last=True,
        shuffle=True, num_workers=4, pin_memory=True
    )
    loader_iter = iter(loader)

    leading_metric, last_best_metric, metric_greater_cmp = {
        "ISC": (torch_fidelity.KEY_METRIC_ISC_MEAN, 0.0, float.__gt__),
        "FID": (torch_fidelity.KEY_METRIC_FID, float("inf"), float.__lt__),
        "KID": (torch_fidelity.KEY_METRIC_KID_MEAN, float("inf"), float.__lt__)
    }[args.leading_metric]

    G = Generator(args.z_size).to(device).train()
    D = Discriminator(not args.disable_sn).to(device).train()
    z_vis = torch.randn(64, args.z_size, device=device)

    optim_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    optim_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))
    
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lambda step: 1.0 - step / args.num_total_steps)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optim_D, lambda step: 1.0 - step / args.num_total_steps)
    
    lr_manager_G = LearningRateManager(optim_G, scheduler_G, args.lr, args.mode, True)
    lr_manager_D = LearningRateManager(optim_D, scheduler_D, args.lr, args.mode, True)

    tb = tensorboard.SummaryWriter(log_dir=args.dir_logs)
    pbar = tqdm.tqdm(total=args.num_total_steps, desc="Training", unit="batch")
    os.makedirs(args.dir_logs, exist_ok=True)

    lr_history = {"step": [], "lr_G_scheduled": [], "lr_G_actual": [], 
                  "lr_D_scheduled": [], "lr_D_actual": []}

    for step in range(args.num_total_steps):
        try:
            real_img, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            real_img, _ = next(loader_iter)
        real_img = real_img.to(device)

        # ====================================================================
        # UPDATE GENERATOR
        # ====================================================================
        G.requires_grad_(True)
        D.requires_grad_(False)
        z = torch.randn(args.batch_size, args.z_size, device=device)
        
        optim_D.zero_grad()
        optim_G.zero_grad()
        
        fake_img = G(z, temperature=args.temperature)
        D_fake = D(fake_img)
        loss_G = hinge_loss_gen(D_fake)

        # 优化：仅在开启阴阳机制时计算一次 D(real_img)，严格隔离计算图
        lr_G_mult = 1.0
        yinyang_enabled = args.mode in {"yinyang_gradscale", "yinyang_daosheng"} and step > 10
    
        if yinyang_enabled:
            with torch.no_grad():
                D_real_for_G = D(real_img)
            lr_G_mult, _ = compute_true_yinyang_lr_scales(D_real_for_G, D_fake, device)
            lr_manager_G.apply_yinyang_scale(lr_G_mult)
            tb.add_scalar("balance/lr_G_mult", lr_G_mult, global_step=step)

        actual_lr_G = lr_manager_G.get_scheduled_lr()
        lr_history["step"].append(step)
        lr_history["lr_G_scheduled"].append(lr_manager_G.get_scheduled_lr())
        lr_history["lr_G_actual"].append(actual_lr_G)

        loss_G.backward()
        optim_G.step()

        # ====================================================================
        # UPDATE DISCRIMINATOR
        # ====================================================================
        G.requires_grad_(False)
        D.requires_grad_(True)
        
        loss_D = None
        for d_iter in range(args.num_dis_updates):
            z = torch.randn(args.batch_size, args.z_size, device=device)
            optim_D.zero_grad()
            optim_G.zero_grad()
            
            fake_img = G(z, temperature=args.temperature)
            D_fake = D(fake_img)
            
            # 优化：D循环内仅执行一次前向传播，结果同时用于 Loss 和 LR缩放
            D_real_for_D = D(real_img)
            loss_D = hinge_loss_dis(D_fake, D_real_for_D)

            lr_D_mult = 1.0
            if yinyang_enabled:
                _, lr_D_mult = compute_true_yinyang_lr_scales(D_real_for_D, D_fake, device)
                lr_manager_D.apply_yinyang_scale(lr_D_mult)
                tb.add_scalar("balance/lr_D_mult", lr_D_mult, global_step=step)

            if d_iter == args.num_dis_updates - 1:
                actual_lr_D = lr_manager_D.get_scheduled_lr()
                lr_history["lr_D_scheduled"].append(lr_manager_D.get_scheduled_lr())
                lr_history["lr_D_actual"].append(actual_lr_D)

            loss_D.backward()
            optim_D.step()

        # ====================================================================
        # Logging
        # ====================================================================
        if (step + 1) % 10 == 0:
            step_info = {"loss_G": loss_G.item(), "loss_D": loss_D.item()}
            pbar.set_postfix(step_info)
            for k, v in step_info.items():
                tb.add_scalar(f"loss/{k}", v, global_step=step)
            tb.add_scalar(f"LR/G_scheduled", lr_manager_G.get_scheduled_lr(), global_step=step)
            tb.add_scalar(f"LR/D_scheduled", lr_manager_D.get_scheduled_lr(), global_step=step)
            
        pbar.update(1)

        lr_manager_G.step_scheduler()
        lr_manager_D.step_scheduler()

        # ====================================================================
        # Evaluation
        # ====================================================================
        next_step = step + 1
        if next_step % args.num_epoch_steps != 0:
            continue

        pbar.close()
        G.eval()

        input1 = TemperatureWrapper(G, args.z_size, args.z_type, num_classes=0, temperature=args.temperature)
        metrics = torch_fidelity.calculate_metrics(
            input1=input1,
            input1_model_num_samples=args.num_samples_for_metrics,
            input2="cifar10-train",
            isc=True, fid=True, kid=True,
        )

        for k, v in metrics.items():
            tb.add_scalar(f"metrics/{k}", v, global_step=next_step)

        samples_vis = G(z_vis, temperature=args.temperature).detach().cpu()
        samples_vis = torchvision.utils.make_grid(samples_vis).permute(1, 2, 0).numpy()
        tb.add_image("observations", samples_vis, global_step=next_step, dataformats="HWC")
        samples_vis = PIL.Image.fromarray(samples_vis)
        samples_vis.save(os.path.join(args.dir_logs, f"{next_step:06d}.png"))

        if metric_greater_cmp(metrics[leading_metric], last_best_metric):
            last_best_metric = metrics[leading_metric]
            dummy_input = torch.zeros(1, args.z_size, device=device)
            torch.jit.save(torch.jit.trace(G, (dummy_input,)), os.path.join(args.dir_logs, "generator.pth"))

        if next_step <= args.num_total_steps:
            pbar = tqdm.tqdm(total=args.num_total_steps, initial=next_step, desc="Training", unit="batch")
            G.train()

    import json
    with open(os.path.join(args.dir_logs, "lr_history.json"), "w") as f:
        lr_history_serializable = {
            k: [float(v_item) if isinstance(v_item, (np.ndarray, torch.Tensor)) else v_item 
                for v_item in v] 
            for k, v in lr_history.items()
        }
        json.dump(lr_history_serializable, f, indent=2)
    print(f"Learning rate history saved to {os.path.join(args.dir_logs, 'lr_history.json')}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    dir = os.getcwd()
    name_map = {
        "baseline": "Baseline",
        "daosheng_temp07": "Daosheng-Temp07",
        "daosheng_temp08": "Daosheng-Temp08",
        "daosheng_temp085": "Daosheng-Temp085",
        "daosheng_temp09": "Daosheng-Temp09",
        "yinyang_gradscale": "True-YinYang-Gradscale",
        "yinyang_daosheng": "True-YinYang-Daosheng",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="baseline", choices=name_map.keys())
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_total_steps", type=int, default=100000)
    parser.add_argument("--num_epoch_steps", type=int, default=5000)
    parser.add_argument("--num_dis_updates", type=int, default=5)
    parser.add_argument("--num_samples_for_metrics", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--z_size", type=int, default=128)
    parser.add_argument("--z_type", type=str, default="normal")
    parser.add_argument("--leading_metric", type=str, default="FID", choices=("ISC", "FID", "KID"))
    parser.add_argument("--disable_sn", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dir_dataset", type=str, default=os.path.join(dir, "dataset"))

    args_partial = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    default_log_dir = os.path.join("logs", f"seed{args_partial.seed}", f"{timestamp}-{name_map[args_partial.mode]}")
    parser.add_argument("--dir_logs", type=str, default=default_log_dir)

    args = parser.parse_args()

    if args.mode == "daosheng_temp07":
        args.temperature = 0.7
    elif args.mode == "daosheng_temp08":
        args.temperature = 0.8
    elif args.mode == "daosheng_temp085":
        args.temperature = 0.85
    elif args.mode == "daosheng_temp09":
        args.temperature = 0.9
    elif args.mode == "yinyang_daosheng":
        args.temperature = 0.85

    set_seed(args.seed)
    print(f"Using random seed: {args.seed}")
    print("Configuration:")
    for k, v in args.__dict__.items():
        print(f"{k:>25}: {v}")

    print("\n" + "="*60)
    print("LEARNING RATE SCHEDULE VERIFICATION")
    print("="*60)
    print(f"Initial LR: {args.lr}")
    print(f"Total Steps: {args.num_total_steps}")
    print(f"Final LR (step {args.num_total_steps}): {args.lr * (1.0 - args.num_total_steps/args.num_total_steps)}")
    print(f"Mode: {args.mode}")
    if args.mode in {"yinyang_gradscale", "yinyang_daosheng"}:
        print("YinYang Mechanism: ENABLED (scales scheduled LR)")
    else:
        print("YinYang Mechanism: DISABLED")
    print("="*60 + "\n")

    train(args)


if __name__ == "__main__":
    main()