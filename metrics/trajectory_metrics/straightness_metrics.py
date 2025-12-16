import torch
import numpy as np
from typing import Dict, List

class TrajectoryMetrics:
    """
    Tools for trajectory straightness evaluation
    """

    @staticmethod
    def path_length_ratio(trajectory: torch.Tensor) -> torch.Tensor:
        """
        Path length ratio
        Straightness = Euclidean distance / Actual path length

        Args:
            trajectory (torch.Tensor): [T, N, 3] T timesteps, N points, 3D coordinates

        Returns:
            torch.Tensor: [1, N] straightness evaluation for every point
        """
        T, N, _ = trajectory.shape
        
        # Euclidean distance from start timestep to end timestep
        euclidean_dist = torch.norm(
            trajectory[-1] - trajectory[0],
            dim = -1
        )
        
        # Distance of straight line from start to end
        path_length = torch.zeros(N, device=trajectory.device)
        
        # Compute the overall path length
        for t in range(T-1):
            step_length = torch.norm(
                trajectory[t+1] - trajectory[t],
                dim=-1
            )
            path_length += step_length
        
        straightness = euclidean_dist / (path_length + 1e-8)
        
        return straightness
    

    @staticmethod
    def direction_consistency(trajectory: torch.Tensor) -> torch.Tensor:
        """
        Direction consistency to evaluate the angle stability

        Args:
            trajectory (torch.Tensor): [T, N, 3] T timesteps, N points, 3D coordinates

        Returns:
            torch.Tensor: (N,) The direction consistency of each point, within the range [-1,1], 
                            where 1 indicates that the directions are completely consistent
        """
        T, N, _ = trajectory.shape
        
        velocities = trajectory[1:] - trajectory[:-1] # (T-1, N, 3)
        
        # normalize the velocities tensor
        velocities_norm = torch.nn.functional.normalize(
            velocities,
            dim=-1,
            eps=1e-8
        ) # (T-1, N, 3)
        
        # Compute the Cosine similarity of adjacent timesteps
        if T < 3:
            return torch.ones(N, device=trajectory.device)
        
        cosine_similarity = (
            velocities_norm[1:] * velocities_norm[:-1]
        ).sum(dim=-1) # (T-2, N)
        
        consistency = cosine_similarity.mean(dim=0) # (N,)
        
        return consistency
    
    @staticmethod
    def average_curvature(trajectory: torch.Tensor) -> torch.Tensor:
        """
        Average Curvature (second derivatives)
        Curvature = || d²x/dt² || / || dx/dt ||²

        Args:
            trajectory (torch.Tensor): (T, N, 3)

        Returns:
            torch.Tensor: Average curvature for each point, 0 is a absolute straight line
        """
        T, N, _ = trajectory.shape
        
        if T < 3:
            return torch.zeros(N, device=trajectory.device)
        
        first_deriv = trajectory[1:] - trajectory[:-1]
        
        second_deriv = first_deriv[1:] - first_deriv[:-1]
        
        acceleration_norm = torch.norm(second_deriv, dim=-1)
        velocity_norm_sq = torch.norm(first_deriv[:-1], dim=-1) ** 2
        
        curvature = acceleration_norm / (velocity_norm_sq + 1e-8)
        
        avg_curvature = curvature.mean(dim=0) # (N,)
        
        return avg_curvature
    
    @staticmethod
    def angular_deviation(trajectory: torch.Tensor) -> torch.Tensor:
        """
        Angular deviation (difference between ideal straight line)

        Args:
            trajectory (torch.Tensor): (T, N, 3)

        Returns:
            torch.Tensor: (N,) average angular deviation(in radian), 0 stands for absolute straight line
        """
        T, N, _ = trajectory.shape
        
        # ideal direction
        ideal_direction = trajectory[-1] - trajectory[0] # (N, 3)
        ideal_direction_norm = torch.nn.functional.normalize(
            ideal_direction, 
            dim=-1,
            eps=1e-8
        )
        
        # practical direction for every single timestep
        velocities = trajectory[1:] - trajectory[:-1]
        velocities_norm = torch.nn.functional.normalize(
            velocities,
            dim=-1,
            eps=1e-8
        )
        
        cosine_sim = (
            velocities_norm * ideal_direction_norm.unsqueeze(0)
        ).sum(dim=-1)
        
        angles = torch.acos(torch.clamp(cosine_sim, -1.0, 1.0)) # (T-1, N)
        
        avg_deviation = angles.mean(dim=0)
        
        return avg_deviation
    
    @staticmethod
    def trajectory_efficiency(
        trajectory: torch.Tensor, 
        clean_target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        方法E: 轨迹效率（去噪进度分析）
        
        Analyzes how efficiently the trajectory approaches the clean target
        
        Args:
            trajectory: (T, N, 3)
            clean_target: (N, 3)
        Returns:
            dict with:
                - progress_per_step: 每步平均进步
                - monotonicity: 单调性（是否一直在接近目标）
        """
        T, N, _ = trajectory.shape
        
        # 计算每个时间步到目标的距离
        distances = []
        for t in range(T):
            dist = torch.norm(trajectory[t] - clean_target, dim=-1)  # (N,)
            distances.append(dist)
        
        distances = torch.stack(distances, dim=0)  # (T, N)
        
        # 每步的进步（负值表示进步）
        progress = distances[1:] - distances[:-1]  # (T-1, N)
        avg_progress = progress.mean(dim=0)  # (N,)
        
        # 单调性：有多少步是在"后退"的
        backward_steps = (progress > 0).float().mean(dim=0)  # (N,)
        monotonicity = 1.0 - backward_steps
        
        return {
            'progress_per_step': avg_progress,
            'monotonicity': monotonicity,
            'final_distance': distances[-1]
        }