from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    seq_len: int = 100
    hidden_dim: int = 128
    flow_dim: int = 256
    num_encoder_layers: int = 2
    num_flow_blocks: int = 4
    num_heads: int = 4
    dropout: float = 0.1

    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 3
    grad_clip_norm: float = 1.0

    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2

    lambda_physical: float = 0.1
    lambda_endpoint: float = 0.5
    lambda_sequence_total: float = 0.5
    lambda_multiscale_total: float = 0.0
    lambda_prefix_cumsum: float = 0.0
    lambda_delta: float = 0.0
    lambda_activity: float = 0.0
    lambda_alarm: float = 0.0
    warmup_epochs_physical: int = 0
    warmup_epochs_prefix: int = 0
    warmup_epochs_delta: int = 0
    warmup_epochs_sequence_total: int = 0
    use_cue: bool = True
    use_activity_gate: bool = False
    cue_dropout_prob: float = 0.0
    guidance_scale: float = 1.0
    activity_threshold: float = 0.01
    activity_sharpness: float = 40.0
    alarm_threshold: float = 0.05
    alarm_sharpness: float = 10.0
    sample_steps: int = 16
    physical_sample_steps: int = 8
    ode_solver: str = "euler"
    physical_ode_solver: str = "euler"
    num_eval_samples: int = 4
    eval_point_estimate_mode: str = "stochastic_mean"
    num_workers: int = 0
    seed: int = 42
    train_sampling: str = "signal_weighted"
    branch_sampling: str = "signal_weighted"
    branch_sampling_temperature: float = 0.8
    augment_noise_std: float = 0.01
    augment_scale_min: float = 0.97
    augment_scale_max: float = 1.03
    checkpoint_metric: str = "sequence_total_loss"
    checkpoint_eval_max_batches: int | None = 32
    checkpoint_eval_use_calibration: bool = False
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    max_train_batches: int | None = 1000
    max_val_batches: int | None = 200
    train_synthetic_mode: str = "all"
    val_synthetic_mode: str = "all"
    test_synthetic_mode: str = "all"
    self_condition: bool = False
    self_condition_prob: float = 0.5
    flow_path: str = "linear"
    flow_sigmoid_bias: float = 0.0
    flow_sigmoid_scale: float = 2.0
    multiscale_windows: tuple[int, ...] = (5, 10, 20, 50)
    endpoint_loss_type: str = "mse"
    endpoint_huber_delta: float = 1.0

    output_dir: Path = Path("outputs")
