from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "data_split"
TRAIN_CSV = DATA_DIR / "train_label.csv"
TEST_CSV = DATA_DIR / "test_label.csv"
TRAIN_IMG_DIR = DATA_DIR / "train_split"
TEST_IMG_DIR = DATA_DIR / "test_split"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"

NUM_CLASSES = 5
IMAGE_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
CLASS_COUNTS = [1534, 317, 850, 164, 249]
TOTAL_TRAIN = sum(CLASS_COUNTS)


@dataclass
class ExpConfig:
    exp_id: int
    name: str

    # architecture
    backbone: str = "resnet50"
    use_gem: bool = False
    gem_p: float = 3.0
    num_outputs: int = 5  # 5 for classification, 1 for regression

    # loss
    loss_type: str = "ce"  # ce | focal | smoothl1
    focal_gamma: float = 2.0
    use_class_weights: bool = False
    label_smoothing: float = 0.0

    # augmentation
    aug_level: int = 0  # 0=none, 1=standard, 2=advanced

    # mixed-sample training
    use_mixup: bool = False
    mixup_alpha: float = 0.4
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0

    # training
    total_epochs: int = 50
    freeze_epochs: int = 5
    batch_size: int = 32
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "step"  # step | cosine
    step_size: int = 30
    step_gamma: float = 0.1
    cosine_t0: int = 10
    cosine_tmult: int = 2
    cosine_eta_min: float = 1e-6

    # SWA
    use_swa: bool = False
    swa_start_epoch: int = 80
    swa_lr: float = 1e-5

    # inference
    use_tta: bool = False
    use_optimized_thresholds: bool = False
    default_thresholds: list[float] = field(
        default_factory=lambda: [0.5, 1.5, 2.5, 3.5]
    )

    # pseudo-labeling
    use_pseudo_labels: bool = False
    pseudo_epochs: int = 10
    pseudo_lr: float = 1e-5
    pseudo_weight: float = 0.5

    # validation split
    val_ratio: float = 0.15
    seed: int = 42

    @property
    def is_regression(self) -> bool:
        return self.num_outputs == 1

    @property
    def exp_name(self) -> str:
        return f"exp{self.exp_id:02d}_{self.name}"

    @property
    def ckpt_dir(self) -> Path:
        return CHECKPOINT_DIR / self.exp_name

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.exp_name


EXPERIMENTS: dict[int, ExpConfig] = {
    0: ExpConfig(exp_id=0, name="baseline"),

    1: ExpConfig(exp_id=1, name="std_aug", aug_level=1),

    2: ExpConfig(exp_id=2, name="adv_aug", aug_level=2),

    3: ExpConfig(
        exp_id=3, name="focal_loss",
        aug_level=2, loss_type="focal", use_class_weights=True,
    ),

    4: ExpConfig(
        exp_id=4, name="label_smooth",
        aug_level=2, loss_type="focal", use_class_weights=True,
        label_smoothing=0.1,
    ),

    5: ExpConfig(
        exp_id=5, name="mixup",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_mixup=True,
    ),

    6: ExpConfig(
        exp_id=6, name="cutmix",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_mixup=True, use_cutmix=True,
    ),

    7: ExpConfig(
        exp_id=7, name="regression",
        aug_level=2, loss_type="smoothl1",
        num_outputs=1,
        use_mixup=True, use_cutmix=True,
    ),

    8: ExpConfig(
        exp_id=8, name="gem",
        aug_level=2, loss_type="smoothl1",
        num_outputs=1,
        use_mixup=True, use_cutmix=True,
        use_gem=True,
    ),

    9: ExpConfig(
        exp_id=9, name="cosine_lr",
        aug_level=2, loss_type="smoothl1",
        num_outputs=1,
        use_mixup=True, use_cutmix=True,
        use_gem=True,
        scheduler="cosine",
    ),

    10: ExpConfig(
        exp_id=10, name="swa",
        aug_level=2, loss_type="smoothl1",
        num_outputs=1,
        use_mixup=True, use_cutmix=True,
        use_gem=True,
        scheduler="cosine",
        use_swa=True,
    ),

    11: ExpConfig(
        exp_id=11, name="tta",
        aug_level=2, loss_type="smoothl1",
        num_outputs=1,
        use_mixup=True, use_cutmix=True,
        use_gem=True,
        scheduler="cosine",
        use_swa=True,
        use_tta=True,
    ),

    12: ExpConfig(
        exp_id=12, name="opt_thresh_opA",
        # Option A: Exp 7 (regression) + threshold optimisation only.
        # Branch point: exp07_regression checkpoint.
        # No GeM / cosine LR / SWA / TTA — isolates threshold gain alone.
        aug_level=2, loss_type="smoothl1",
        num_outputs=1,
        use_mixup=True, use_cutmix=True,
        use_optimized_thresholds=True,
    ),

    13: ExpConfig(
        exp_id=13, name="pseudo_label",
        aug_level=2, loss_type="smoothl1",
        num_outputs=1,
        use_mixup=True, use_cutmix=True,
        use_gem=True,
        scheduler="cosine",
        use_swa=True,
        use_tta=True,
        use_optimized_thresholds=True,
        use_pseudo_labels=True,
    ),

    14: ExpConfig(
        exp_id=14, name="ensemble",
        aug_level=2, loss_type="smoothl1",
        num_outputs=1,
        use_mixup=True, use_cutmix=True,
        use_gem=True,
        scheduler="cosine",
        use_swa=True,
        use_tta=True,
        use_optimized_thresholds=True,
        use_pseudo_labels=True,
    ),
}


def get_config(exp_id: int) -> ExpConfig:
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment id: {exp_id}. Valid: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[exp_id]
