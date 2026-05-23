from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from posefork.tasks.g1_arm_reach.agents.mode_actor_critic import ModeConditionedActorCritic
from posefork.algorithms.mode_ppo import ModePPO

import rsl_rl.runners.on_policy_runner as _rsl_on_policy_runner

_rsl_on_policy_runner.ModeConditionedActorCritic = ModeConditionedActorCritic
_rsl_on_policy_runner.ModePPO = ModePPO


@configclass
class G1ArmReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "yb_g1_arm_reach"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1ArmReachSingleAPPORunnerCfg(G1ArmReachPPORunnerCfg):
    experiment_name = "yb_g1_arm_reach_single_A"  # phi=-1.15


@configclass
class G1ArmReachSingleBPPORunnerCfg(G1ArmReachPPORunnerCfg):
    experiment_name = "yb_g1_arm_reach_single_B"  # phi=-0.80


@configclass
class G1ArmReachModeHeadPPORunnerCfg(G1ArmReachPPORunnerCfg):
    experiment_name = "yb_g1_arm_reach_mode_head"
    policy = RslRlPpoActorCriticCfg(
        class_name="ModeConditionedActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="ModePPO",
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=0.3,
    )
