"""
Phase 2: PPO Model Implementation for Scheduling

This package contains the PPO (Proximal Policy Optimization) model implementation
for the scheduling game environment. All components are designed to handle
variable-sized inputs (10-1000+ jobs) without architectural changes.

Components:
- state_encoder: Converts raw observations to structured features
- transformer_policy: Attention-based policy network
- action_masking: Handles invalid action filtering  
- ppo_scheduler: Core PPO algorithm
- rollout_buffer: Experience storage
- curriculum: Progressive training stages
- train: Main training loop
- evaluate: Model evaluation
"""