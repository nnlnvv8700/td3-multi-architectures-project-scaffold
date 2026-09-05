"""
批量实验运行器
==============
自动运行多个架构的对比实验，支持并行/串行执行，自动汇总结果。

使用示例：
    # 运行所有架构
    python training/run_all_experiments.py --env KukaIiwa7Track-v0
    
    # 运行特定架构
    python training/run_all_experiments.py --env KukaIiwa7Track-v0 --architectures mlp gnn
    
    # 并行运行（实验性，需要多GPU）
    python training/run_all_experiments.py --env KukaIiwa7Track-v0 --parallel --gpus 0 1

作者: Auto-generated
日期: 2025-11-10
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Optional
import json
import time

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class ExperimentRunner:
    """批量实验运行管理器"""
    
    def __init__(
        self,
        env: str = "KukaIiwa7Track-v0",
        architectures: Optional[List[str]] = None,
        max_timesteps: int = 200000,
        num_runs_per_arch: int = 1,
        parallel: bool = False,
        gpus: Optional[List[int]] = None,
        base_save_dir: str = "./results",
        timeout: float = 7200,
    ):
        """
        初始化实验运行器
        
        Args:
            env: 环境名称
            architectures: 要测试的架构列表，None表示全部
            max_timesteps: 每次实验的最大步数
            num_runs_per_arch: 每个架构运行的次数（用于统计稳定性）
            parallel: 是否并行运行（需要多GPU）
            gpus: GPU设备列表
            base_save_dir: 结果保存根目录
        """
        self.env = env
        self.architectures = architectures or ["mlp", "gnn", "transformer", "gnn_transformer"]
        self.max_timesteps = max_timesteps
        self.num_runs_per_arch = num_runs_per_arch
        self.parallel = parallel
        self.gpus = gpus or [0]
        self.base_save_dir = base_save_dir
        self.base_seed = 42
        self.timeout = timeout
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_timesteps <= 0 or num_runs_per_arch <= 0:
            raise ValueError("max_timesteps and num_runs_per_arch must be positive")
        if len(set(self.gpus)) != len(self.gpus) or any(gpu < 0 for gpu in self.gpus):
            raise ValueError("gpus must contain distinct non-negative indices")
        
        # 实验记录
        self.experiment_log: List[Dict] = []
        self.failed_experiments: List[Dict] = []
        
    def run_single_experiment(
        self,
        architecture: str,
        run_id: int = 1,
        gpu_id: int = 0
    ) -> Dict:
        """
        运行单个实验
        
        Args:
            architecture: Actor架构类型
            run_id: 运行编号（同一架构的第几次运行）
            gpu_id: 使用的GPU ID
            
        Returns:
            实验结果字典，包含状态、耗时等信息
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting Experiment: {architecture.upper()} (Run {run_id}/{self.num_runs_per_arch})")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # 构建命令
        train_script = os.path.join(PROJECT_ROOT, "training", "train_experiment.py")
        cmd = [
            sys.executable,  # 使用当前Python解释器
            train_script,
            "--env", self.env,
            "--actor_arch", architecture,
            "--max_timesteps", str(self.max_timesteps),
            "--save_dir", self.base_save_dir,
            "--seed", str(self.base_seed + run_id - 1),
        ]
        
        # 设置环境变量（GPU）
        env_vars = os.environ.copy()
        env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env_vars["PYTHONIOENCODING"] = "utf-8"
        
        # 运行实验
        try:
            result = subprocess.run(
                cmd,
                env=env_vars,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout
            )
            
            success = result.returncode == 0
            elapsed_time = time.time() - start_time
            
            experiment_result = {
                "architecture": architecture,
                "run_id": run_id,
                "gpu_id": gpu_id,
                "success": success,
                "elapsed_time": elapsed_time,
                "return_code": result.returncode,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if success:
                print(f"✅ SUCCESS: {architecture} (Run {run_id}) completed in {elapsed_time/60:.1f} minutes")
            else:
                print(f"❌ FAILED: {architecture} (Run {run_id})")
                print(f"Error output:\n{result.stderr[-500:]}")  # 打印最后500字符
                experiment_result["error"] = result.stderr[-500:]
                
            return experiment_result
            
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            print(f"⏰ TIMEOUT: {architecture} (Run {run_id}) exceeded {self.timeout} seconds")
            return {
                "architecture": architecture,
                "run_id": run_id,
                "gpu_id": gpu_id,
                "success": False,
                "elapsed_time": elapsed_time,
                "error": f"Timeout after {self.timeout} seconds",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"💥 EXCEPTION: {architecture} (Run {run_id}) - {str(e)}")
            return {
                "architecture": architecture,
                "run_id": run_id,
                "gpu_id": gpu_id,
                "success": False,
                "elapsed_time": elapsed_time,
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def run_all_sequential(self) -> None:
        """串行运行所有实验"""
        print(f"\n{'#'*60}")
        print(f"# Batch Experiment Runner - Sequential Mode")
        print(f"# Environment: {self.env}")
        print(f"# Architectures: {', '.join(self.architectures)}")
        print(f"# Runs per architecture: {self.num_runs_per_arch}")
        print(f"# Total experiments: {len(self.architectures) * self.num_runs_per_arch}")
        print(f"{'#'*60}\n")
        
        total_start = time.time()
        
        for arch in self.architectures:
            for run_id in range(1, self.num_runs_per_arch + 1):
                result = self.run_single_experiment(arch, run_id, gpu_id=self.gpus[0])
                self.experiment_log.append(result)
                
                if not result["success"]:
                    self.failed_experiments.append(result)
        
        total_elapsed = time.time() - total_start
        self._print_summary(total_elapsed)
        self._save_experiment_log()
    
    def run_all_parallel(self) -> None:
        """并行运行所有实验（需要多GPU）"""
        print(f"\n{'#'*60}")
        print(f"# Batch Experiment Runner - Parallel Mode")
        print(f"# Available GPUs: {self.gpus}")
        print(f"# WARNING: Parallel mode is experimental!")
        print(f"{'#'*60}\n")
        
        # 简化实现：使用 concurrent.futures
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        total_start = time.time()
        
        # 准备所有任务
        tasks = []
        for arch in self.architectures:
            for run_id in range(1, self.num_runs_per_arch + 1):
                gpu_id = self.gpus[len(tasks) % len(self.gpus)]
                tasks.append((arch, run_id, gpu_id))
        
        # 并行执行
        # One sequential queue per GPU. A shared queue can accidentally schedule
        # two preassigned jobs on the same GPU when another job finishes first.
        with ThreadPoolExecutor(max_workers=len(self.gpus)) as executor:
            future_to_task = {
                executor.submit(self._run_gpu_queue, [task for task in tasks if task[2] == gpu]): gpu
                for gpu in self.gpus
            }
            
            for future in as_completed(future_to_task):
                for result in future.result():
                    self.experiment_log.append(result)
                    if not result["success"]:
                        self.failed_experiments.append(result)
        
        total_elapsed = time.time() - total_start
        self._print_summary(total_elapsed)
        self._save_experiment_log()

    def _run_gpu_queue(self, tasks):
        return [self.run_single_experiment(arch, run_id, gpu) for arch, run_id, gpu in tasks]
    
    def _print_summary(self, total_elapsed: float) -> None:
        """打印实验汇总"""
        print(f"\n{'='*60}")
        print(f"📊 EXPERIMENT SUMMARY")
        print(f"{'='*60}\n")
        
        total_experiments = len(self.experiment_log)
        successful = sum(1 for exp in self.experiment_log if exp["success"])
        failed = len(self.failed_experiments)
        
        print(f"Total experiments: {total_experiments}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️  Total time: {total_elapsed/60:.1f} minutes")
        print(f"⏱️  Average time per experiment: {total_elapsed/max(1, total_experiments)/60:.1f} minutes")
        
        # 按架构统计
        print(f"\n{'─'*60}")
        print("Per-architecture results:")
        print(f"{'─'*60}")
        
        for arch in self.architectures:
            arch_results = [exp for exp in self.experiment_log if exp["architecture"] == arch]
            arch_success = sum(1 for exp in arch_results if exp["success"])
            avg_time = sum(exp["elapsed_time"] for exp in arch_results) / len(arch_results) if arch_results else 0
            
            status = "✅" if arch_success == len(arch_results) else "⚠️"
            print(f"{status} {arch.upper():20s}: {arch_success}/{len(arch_results)} successful, "
                  f"avg {avg_time/60:.1f} min")
        
        # 失败详情
        if self.failed_experiments:
            print(f"\n{'─'*60}")
            print("❌ Failed experiments details:")
            print(f"{'─'*60}")
            for exp in self.failed_experiments:
                print(f"  • {exp['architecture']} (Run {exp['run_id']}): {exp.get('error', 'Unknown error')[:100]}")
        
        print(f"\n{'='*60}\n")
    
    def _save_experiment_log(self) -> None:
        """保存实验日志"""
        log_dir = os.path.join(self.base_save_dir, "batch_experiments")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"experiment_log_{timestamp}.json")
        
        log_data = {
            "config": {
                "env": self.env,
                "architectures": self.architectures,
                "max_timesteps": self.max_timesteps,
                "num_runs_per_arch": self.num_runs_per_arch,
                "parallel": self.parallel,
                "gpus": self.gpus
            },
            "experiments": self.experiment_log,
            "summary": {
                "total": len(self.experiment_log),
                "successful": sum(1 for exp in self.experiment_log if exp["success"]),
                "failed": len(self.failed_experiments)
            }
        }
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📝 Experiment log saved to: {log_file}")
    
    def run(self) -> None:
        """执行所有实验"""
        if self.parallel and len(self.gpus) > 1:
            self.run_all_parallel()
        else:
            self.run_all_sequential()


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="批量运行多个架构的TD3实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行所有架构（串行）
  python training/run_all_experiments.py --env KukaIiwa7Track-v0
  
  # 只运行MLP和GNN
  python training/run_all_experiments.py --env KukaIiwa7Track-v0 --architectures mlp gnn
  
  # 每个架构运行3次（统计稳定性）
  python training/run_all_experiments.py --env KukaIiwa7Track-v0 --num_runs 3
  
  # 并行运行（需要多GPU）
  python training/run_all_experiments.py --env KukaIiwa7Track-v0 --parallel --gpus 0 1 2 3
        """
    )
    
    parser.add_argument(
        "--env",
        type=str,
        default="KukaIiwa7Track-v0",
        help="环境名称 (default: KukaIiwa7Track-v0)"
    )
    
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=["mlp", "gnn", "transformer", "gnn_transformer"],
        default=None,
        help="要测试的架构列表，不指定则运行全部 (choices: mlp, gnn, transformer, gnn_transformer)"
    )
    
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=200000,
        help="每次实验的最大训练步数 (default: 200000)"
    )
    
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="每个架构运行的次数，用于统计稳定性 (default: 1)"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="启用并行模式（需要多GPU，实验性功能）"
    )
    
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0],
        help="可用的GPU列表 (default: [0])"
    )
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="结果保存目录 (default: ./results)"
    )
    
    parser.add_argument("--timeout", type=float, default=7200,
                        help="Maximum seconds per experiment (default: 7200)")
    args = parser.parse_args()
    
    # 创建运行器
    runner = ExperimentRunner(
        env=args.env,
        architectures=args.architectures,
        max_timesteps=args.max_timesteps,
        num_runs_per_arch=args.num_runs,
        parallel=args.parallel,
        gpus=args.gpus,
        base_save_dir=args.save_dir,
        timeout=args.timeout,
    )
    
    # 执行实验
    try:
        runner.run()
        if runner.failed_experiments:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
        runner._print_summary(0)
        runner._save_experiment_log()
        sys.exit(1)


if __name__ == "__main__":
    main()
