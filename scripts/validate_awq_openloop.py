# -*- coding: utf-8 -*-
"""
验证 AWQ 量化后的 Wall-OSS 模型开环控制效果
基于 draw_openloop_plot.py 逻辑修改

使用方法:
    # 验证 AWQ 量化模型
    /home/xieqijia/tools/anaconda3/envs/wallx-AngelSlim/bin/python scripts/validate_awq_openloop.py \
        --quantized_model_path ./output/wall_oss_int4_awq/wall_oss_int4_awq \
        --num_samples 200 \
        --gpu 4

    # 对比原始模型和量化模型
    /home/xieqijia/tools/anaconda3/envs/wallx-AngelSlim/bin/python scripts/validate_awq_openloop.py \
        --quantized_model_path ./output/wall_oss_int4_awq/wall_oss_int4_awq \
        --compare_with_original \
        --num_samples 200 \
        --gpu 4
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加 Wall-OSS 路径
sys.path.insert(0, "/home/xieqijia/Project/WALL-OSS/wall-x_eager_attention")

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.data.load_lerobot_dataset import load_test_dataset, get_data_configs

# 默认配置路径
CONFIG_PATH = "/home/xieqijia/Project/WALL-OSS/wall-x/workspace/libero/config_qact.yml"
ACTION_TOKENIZER_PATH = "/home/xieqijia/Models/fast"
ORIGINAL_MODEL_PATH = "/home/xieqijia/Project/WALL-OSS/wall-x/workspace/libero/workspace/finetuned_new"


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["data"]["model_type"] = config.get("model_type")
    return config


def load_original_model(model_path, config, action_tokenizer_path, gpu_id=0):
    """加载原始模型 (BF16)"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model = Qwen2_5_VLMoEForAction.from_pretrained(
        model_path,
        train_config=config,
        action_tokenizer_path=action_tokenizer_path,
        use_head_dim_padding=False,
    )
    model.eval()
    model = model.to("cuda")
    model = model.bfloat16()
    return model


def load_awq_quantized_model(quantized_model_path, config, action_tokenizer_path, gpu_id=0):
    """加载 AWQ 量化后的模型

    AWQ 量化模型使用 HuggingFace Transformers 格式保存，可以直接用 from_pretrained 加载
    但需要确保配置文件中的 quantization_config 正确设置
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"Loading AWQ quantized model from: {quantized_model_path}")

    # AWQ 模型使用 AutoAWQ 或 transformers 加载
    # 由于 AWQ 保存的是伪量化后的权重 (float16 with quantization config),
    # 我们可以直接加载，推理时会使用 WQLinearGEMM 进行反量化计算
    model = Qwen2_5_VLMoEForAction.from_pretrained(
        quantized_model_path,
        train_config=config,
        action_tokenizer_path=action_tokenizer_path,
        use_head_dim_padding=False,
        torch_dtype=torch.float16,  # AWQ 使用 float16
    )
    model.eval()
    model = model.to("cuda")

    return model


def run_openloop_inference(model, dataloader, config, lerobot_config, pred_horizon, origin_action_dim):
    """运行开环控制推理

    类似于 draw_openloop_plot.py 的逻辑:
    - 遍历数据集中的所有帧
    - 每 pred_horizon 步进行一次预测
    - 填充完整的轨迹
    """
    total_frames = len(dataloader)
    predict_mode = "fast" if config.get("use_fast_tokenizer", False) else "diffusion"
    action_dim = 20 if predict_mode == "diffusion" else origin_action_dim

    # 初始化轨迹张量
    gt_traj = torch.zeros((total_frames, origin_action_dim))
    pred_traj = torch.zeros((total_frames, origin_action_dim))

    for idx, batch in tqdm(enumerate(dataloader), total=total_frames, desc="Open-loop inference"):
        # 每 pred_horizon 步进行一次预测
        if idx % pred_horizon == 0 and idx + pred_horizon < total_frames:
            batch = batch.to("cuda")
            with torch.no_grad():
                outputs = model(
                    **batch,
                    action_dim=action_dim,
                    pred_horizon=pred_horizon,
                    mode="predict",
                    predict_mode=predict_mode,
                )
                pred_traj[idx : idx + pred_horizon] = (
                    outputs["predict_action"][:, :, :origin_action_dim]
                    .detach()
                    .cpu()
                    .squeeze(0)
                )

            # 反归一化真值动作
            gt_action_chunk = batch["action_chunk"][:, :, :origin_action_dim]
            dof_mask = batch["dof_mask"].to(gt_action_chunk.dtype)
            denormalized_gt = (
                model.action_preprocessor.normalizer_action.unnormalize_data(
                    gt_action_chunk,
                    [lerobot_config.get("repo_id", "lerobot/libero_goal_image")],
                    dof_mask,
                ).squeeze(0)
            )
            gt_traj[idx : idx + pred_horizon] = denormalized_gt.detach().cpu()

    return gt_traj.numpy(), pred_traj.numpy()


def compute_metrics(pred, gt, label="Model"):
    """计算评估指标"""
    mse = np.mean((pred - gt) ** 2)
    mae = np.mean(np.abs(pred - gt))
    rmse = np.sqrt(mse)

    # 每个维度的指标
    mse_per_dim = np.mean((pred - gt) ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(pred - gt), axis=0)

    return {
        f"{label}_mse": mse,
        f"{label}_mae": mae,
        f"{label}_rmse": rmse,
        f"{label}_mse_per_dim": mse_per_dim,
        f"{label}_mae_per_dim": mae_per_dim,
    }


def compute_quantization_metrics(pred_orig, pred_quant):
    """计算原始模型和量化模型之间的差异"""
    diff = pred_quant - pred_orig

    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_diff = np.max(np.abs(diff))

    # 余弦相似度
    cos_sim = np.dot(pred_orig.flatten(), pred_quant.flatten()) / (
        np.linalg.norm(pred_orig.flatten()) * np.linalg.norm(pred_quant.flatten())
    )

    return {
        "diff_mse": mse,
        "diff_mae": mae,
        "diff_max": max_diff,
        "cosine_similarity": cos_sim,
    }


def plot_comparison(gt_traj, pred_orig, pred_quant, save_dir, num_samples=200):
    """绘制对比图"""
    # 限制样本数量用于可视化
    if num_samples > 0:
        gt_traj = gt_traj[:num_samples]
        pred_orig = pred_orig[:num_samples]
        pred_quant = pred_quant[:num_samples]

    timesteps = gt_traj.shape[0]
    origin_action_dim = gt_traj.shape[1]

    # 图1: 对比真值和两个模型的预测
    fig1, axs1 = plt.subplots(
        origin_action_dim, 1, figsize=(15, 5 * origin_action_dim), sharex=True
    )
    fig1.suptitle("AWQ Quantization Comparison: GT vs Predictions", fontsize=16)

    if origin_action_dim == 1:
        axs1 = [axs1]

    for i in range(origin_action_dim):
        axs1[i].plot(range(timesteps), gt_traj[:, i], label="Ground Truth", color='black', linewidth=2)
        axs1[i].plot(range(timesteps), pred_orig[:, i], label="Original Model (BF16)", color='blue', linestyle='--', alpha=0.7)
        axs1[i].plot(range(timesteps), pred_quant[:, i], label="AWQ Quantized (INT4)", color='red', linestyle=':', alpha=0.7)
        axs1[i].set_ylabel(f"Action Dim {i+1}")
        axs1[i].legend(loc='upper right')
        axs1[i].grid(True)

    axs1[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path1 = os.path.join(save_dir, "awq_comparison_gt_vs_preds.png")
    plt.savefig(save_path1, dpi=150)
    print(f"Saved comparison plot to {save_path1}")
    plt.close()

    # 图2: 两个预测之间的差异
    diff = pred_quant - pred_orig
    fig2, axs2 = plt.subplots(
        origin_action_dim, 1, figsize=(15, 5 * origin_action_dim), sharex=True
    )
    fig2.suptitle("Prediction Difference (AWQ - Original)", fontsize=16)

    for i in range(origin_action_dim):
        axs2[i].plot(range(timesteps), diff[:, i], color='green')
        axs2[i].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axs2[i].fill_between(range(timesteps), diff[:, i], 0, alpha=0.3, color='green')
        axs2[i].set_ylabel(f"Diff Dim {i+1}")
        axs2[i].grid(True)

    axs2[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path2 = os.path.join(save_dir, "awq_prediction_diff.png")
    plt.savefig(save_path2, dpi=150)
    print(f"Saved difference plot to {save_path2}")
    plt.close()


def plot_single_model_results(gt_traj, pred_traj, save_dir, model_name="AWQ", num_samples=200):
    """绘制单一模型的结果"""
    if num_samples > 0:
        gt_traj = gt_traj[:num_samples]
        pred_traj = pred_traj[:num_samples]

    timesteps = gt_traj.shape[0]
    origin_action_dim = gt_traj.shape[1]

    fig, axs = plt.subplots(
        origin_action_dim, 1, figsize=(15, 5 * origin_action_dim), sharex=True
    )
    fig.suptitle(f"{model_name} Model: GT vs Prediction", fontsize=16)

    if origin_action_dim == 1:
        axs = [axs]

    for i in range(origin_action_dim):
        axs[i].plot(range(timesteps), gt_traj[:, i], label="Ground Truth", color='black', linewidth=2)
        axs[i].plot(range(timesteps), pred_traj[:, i], label=f"{model_name}", color='blue', linestyle='--', alpha=0.7)
        axs[i].set_ylabel(f"Action Dim {i+1}")
        axs[i].legend(loc='upper right')
        axs[i].grid(True)

    axs[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f"{model_name.lower()}_results.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved {model_name} results plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="验证 AWQ 量化模型的开环控制效果")
    parser.add_argument("--quantized_model_path", type=str, required=True,
                        help="AWQ 量化模型的路径 (HuggingFace 格式)")
    parser.add_argument("--compare_with_original", action="store_true",
                        help="是否同时运行原始模型进行对比")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="用于可视化的样本数量 (0 表示全部)")
    parser.add_argument("--pred_horizon", type=int, default=32,
                        help="预测步长")
    parser.add_argument("--origin_action_dim", type=int, default=7,
                        help="原始动作维度")
    parser.add_argument("--gpu", type=int, default=4,
                        help="使用的 GPU ID")
    parser.add_argument("--save_dir", type=str, default="./output/awq_validation",
                        help="保存结果的目录")
    args = parser.parse_args()

    origin_action_dim = args.origin_action_dim
    pred_horizon = args.pred_horizon
    num_samples = args.num_samples
    gpu_id = args.gpu

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载配置
    print("=" * 60)
    print("Loading configuration...")
    print("=" * 60)
    config = load_config(CONFIG_PATH)

    # 获取测试数据加载器
    print("\nLoading test dataset...")
    dataload_config = get_data_configs(config["data"])
    lerobot_config = dataload_config.get("lerobot_config", {})
    dataset = load_test_dataset(config, lerobot_config, seed=42)
    dataloader = dataset.get_dataloader()
    total_frames = len(dataloader)
    print(f"Total frames in dataset: {total_frames}")

    # ============================================================
    # 运行量化模型推理
    # ============================================================
    print("\n" + "=" * 60)
    print("Loading AWQ quantized model...")
    print("=" * 60)
    model_quant = load_awq_quantized_model(
        args.quantized_model_path, config, ACTION_TOKENIZER_PATH, gpu_id
    )

    print("\nRunning inference with AWQ quantized model...")
    gt_traj, pred_traj_quant = run_openloop_inference(
        model_quant, dataloader, config, lerobot_config, pred_horizon, origin_action_dim
    )

    # 计算量化模型的指标
    metrics_quant = compute_metrics(pred_traj_quant, gt_traj, "AWQ")

    # 释放量化模型内存
    del model_quant
    torch.cuda.empty_cache()

    # ============================================================
    # 如果需要，运行原始模型进行对比
    # ============================================================
    if args.compare_with_original:
        # 重新创建数据加载器
        dataset = load_test_dataset(config, lerobot_config, seed=42)
        dataloader = dataset.get_dataloader()

        print("\n" + "=" * 60)
        print("Loading original model (BF16)...")
        print("=" * 60)
        model_orig = load_original_model(
            ORIGINAL_MODEL_PATH, config, ACTION_TOKENIZER_PATH, gpu_id
        )

        print("\nRunning inference with original model...")
        _, pred_traj_orig = run_openloop_inference(
            model_orig, dataloader, config, lerobot_config, pred_horizon, origin_action_dim
        )

        # 计算原始模型的指标
        metrics_orig = compute_metrics(pred_traj_orig, gt_traj, "Original")

        # 计算量化差异指标
        metrics_diff = compute_quantization_metrics(pred_traj_orig, pred_traj_quant)

        # 释放原始模型内存
        del model_orig
        torch.cuda.empty_cache()

        # ============================================================
        # 打印对比结果
        # ============================================================
        print("\n" + "=" * 60)
        print("Comparison Results")
        print("=" * 60)

        print("\n【Ground Truth Metrics】")
        print(f"Original Model MSE: {metrics_orig['Original_mse']:.6f}")
        print(f"Original Model MAE: {metrics_orig['Original_mae']:.6f}")
        print(f"Original Model RMSE: {metrics_orig['Original_rmse']:.6f}")
        print(f"  Per-dim MSE: {metrics_orig['Original_mse_per_dim']}")

        print(f"\nAWQ Quantized Model MSE: {metrics_quant['AWQ_mse']:.6f}")
        print(f"AWQ Quantized Model MAE: {metrics_quant['AWQ_mae']:.6f}")
        print(f"AWQ Quantized Model RMSE: {metrics_quant['AWQ_rmse']:.6f}")
        print(f"  Per-dim MSE: {metrics_quant['AWQ_mse_per_dim']}")

        print("\n【Quantization Difference Metrics】")
        print(f"MSE between Original and AWQ: {metrics_diff['diff_mse']:.6f}")
        print(f"MAE between Original and AWQ: {metrics_diff['diff_mae']:.6f}")
        print(f"Max absolute difference: {metrics_diff['diff_max']:.6f}")
        print(f"Cosine similarity: {metrics_diff['cosine_similarity']:.6f}")

        # ============================================================
        # 绘制对比图
        # ============================================================
        print("\nGenerating comparison plots...")
        plot_comparison(gt_traj, pred_traj_orig, pred_traj_quant, args.save_dir, num_samples)

    else:
        # 只打印量化模型的结果
        print("\n" + "=" * 60)
        print("AWQ Quantized Model Results")
        print("=" * 60)
        print(f"MSE vs Ground Truth: {metrics_quant['AWQ_mse']:.6f}")
        print(f"MAE vs Ground Truth: {metrics_quant['AWQ_mae']:.6f}")
        print(f"RMSE vs Ground Truth: {metrics_quant['AWQ_rmse']:.6f}")
        print(f"Per-dim MSE: {metrics_quant['AWQ_mse_per_dim']}")
        print(f"Per-dim MAE: {metrics_quant['AWQ_mae_per_dim']}")

        # 绘制单一模型结果
        print("\nGenerating results plot...")
        plot_single_model_results(gt_traj, pred_traj_quant, args.save_dir, "AWQ", num_samples)

    print("\n" + "=" * 60)
    print(f"Done! Results saved to: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
