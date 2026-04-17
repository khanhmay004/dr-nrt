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

# IDRiD supplement (Phase C)
IDRID_DIR = ROOT_DIR / "B_Disease_Grading"
IDRID_PROCESSED_DIR = ROOT_DIR / "data" / "idrid_processed"
IDRID_CSV = ROOT_DIR / "data" / "idrid_labels.csv"

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
    loss_type: str = "ce"  # ce | focal | smoothl1 | corn | cumlink | emd
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

    # sampling
    use_weighted_sampler: bool = False

    # offline oversampling
    oversample_target: int = 0  # 0 = disabled; >0 = target count per class
    oversample_dir: str = ""  # path to oversampled images folder

    # contrastive pre-training
    use_contrastive_pretrain: bool = False
    contrastive_epochs: int = 50
    contrastive_lr: float = 1e-3
    contrastive_temperature: float = 0.07
    contrastive_proj_dim: int = 128
    contrastive_data: str = "aptos"  # aptos | eyepacs
    eyepacs_dir: str = ""
    eyepacs_csv: str = ""

    # joint contrastive fine-tuning (OrdSupCon as auxiliary loss during supervised training)
    use_joint_contrastive: bool = False
    joint_contrastive_weight: float = 0.1
    joint_contrastive_warmup: int = 0  # epochs after unfreeze to ramp λ from 0; 0=disabled
    detach_contrastive_backbone: bool = False  # stop contrastive gradient to backbone

    # IDRiD supplement (Phase C)
    use_idrid_supplement: bool = False
    idrid_processed_dir: str = ""
    idrid_csv: str = ""

    # head regularization
    head_dropout: float = 0.0  # dropout probability before FC (0.0 = disabled)

    # checkpoint to load backbone from (for eval-only or contrastive init)
    load_checkpoint: str = ""

    # pre-trained backbone to load for fine-tuning (skip contrastive Stage 1)
    load_backbone: str = ""

    # layer-wise LR decay for discriminative fine-tuning (0.0 = disabled)
    layerwise_lr_decay: float = 0.0

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
        # Branched from Exp 3 (focal loss, classification). Adds GeM pooling.
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
    ),

    9: ExpConfig(
        exp_id=9, name="cosine_lr",
        # Exp 8 + cosine annealing LR scheduler.
        aug_level=2, loss_type="focal", use_class_weights=True,
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
        # Exp 9 + pseudo-labeling. Classification, focal loss, no regression.
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        scheduler="cosine",
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

    # === Phase A: Ordinal Contrastive Experiments (docs/03-ordinal-supcon.md) ===

    # A0: Exp 8 baseline + ECE (eval only — load checkpoint)
    100: ExpConfig(
        exp_id=100, name="a0_baseline_ece",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        load_checkpoint=str(CHECKPOINT_DIR / "exp08_gem" / "exp08_gem_best.pth"),
    ),

    # A0b: Exp 8 + WeightedRandomSampler
    101: ExpConfig(
        exp_id=101, name="a0b_weighted_sampler",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        use_weighted_sampler=True,
    ),

    # A0c-v2: Exp 8 + Offline Oversampling (target 1000, Level 1.5 aug)
    102: ExpConfig(
        exp_id=102, name="a0c_offline_oversample",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # A1-v2: OrdSupCon pre-train on APTOS → fine-tune on APTOS
    # Fixes vs v1: offline oversample instead of WRS, 30 contrastive epochs,
    # freeze_epochs=2, VRAM cleared between stages
    103: ExpConfig(
        exp_id=103, name="a1_ordsupcon_aptos",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        use_contrastive_pretrain=True,
        contrastive_epochs=30,
        contrastive_lr=1e-3,
        contrastive_temperature=0.07,
        contrastive_proj_dim=128,
        contrastive_data="aptos",
        total_epochs=60,
        freeze_epochs=2,
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # === Phase B: EyePACS Contrastive Pre-training (docs/03-ordinal-supcon.md §10) ===

    # A2: OrdSupCon pre-train on EyePACS (35K) → fine-tune on APTOS + offline oversample
    200: ExpConfig(
        exp_id=200, name="a2_ordsupcon_eyepacs",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        use_contrastive_pretrain=True,
        contrastive_epochs=20,
        contrastive_lr=1e-3,
        contrastive_temperature=0.07,
        contrastive_proj_dim=128,
        contrastive_data="eyepacs",
        eyepacs_dir=str(ROOT_DIR / "data" / "eyepacs_processed" / "eyepacs_processed"),  # nested directory
        eyepacs_csv=str(ROOT_DIR / "data" / "eyepacs_processed" / "eyepacs_labels.csv"),
        total_epochs=60,
        freeze_epochs=2,
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # === Phase D: Regularization & LR Schedule (docs/03-ordinal-supcon.md §11.4) ===

    # D1: A0 recipe + Dropout(0.3) before FC + CosineAnnealingLR + 100 epochs + wd=1e-4
    300: ExpConfig(
        exp_id=300, name="d1_dropout_cosine",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        total_epochs=80,
        freeze_epochs=5,
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # A2-v2: Reuse A2 EyePACS backbone, retrain fine-tune with freeze_epochs=5
    # Tests hypothesis: domain gap is a BN re-calibration issue
    201: ExpConfig(
        exp_id=201, name="a2v2_freeze5_eyepacs",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        load_backbone=str(
            CHECKPOINT_DIR / "exp200_a2_ordsupcon_eyepacs" / "exp200_a2_ordsupcon_eyepacs_backbone.pth"
        ),
        total_epochs=60,
        freeze_epochs=7,
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # === Phase F: Joint Contrastive Fine-tuning (docs/03-ordinal-supcon.md §11.6) ===

    # F2: Joint Focal + OrdSupCon, D1 regularization recipe
    501: ExpConfig(
        exp_id=501, name="f2_joint_ordsupcon",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        batch_size=24,
        total_epochs=80,
        freeze_epochs=7,
        lr_finetune=5e-5,
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
        use_joint_contrastive=True,
        joint_contrastive_weight=0.1,
        contrastive_temperature=0.07,
        contrastive_proj_dim=128,
    ),

    # F3: Fix confounds — D1 exact hyperparams + joint OrdSupCon
    502: ExpConfig(
        exp_id=502, name="f3_joint_fixed",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        batch_size=32,              # match D1 (was 24 in F2)
        total_epochs=80,
        freeze_epochs=5,            # match D1 (was 7 in F2)
        lr_finetune=1e-4,           # match D1 (was 5e-5 in F2)
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
        use_joint_contrastive=True,
        joint_contrastive_weight=0.1,
        contrastive_temperature=0.07,
        contrastive_proj_dim=128,
    ),

    # F4: Low λ + warmup — let classification converge first
    503: ExpConfig(
        exp_id=503, name="f4_joint_warmup",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        batch_size=32,
        total_epochs=80,
        freeze_epochs=5,
        lr_finetune=1e-4,
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
        use_joint_contrastive=True,
        joint_contrastive_weight=0.01,   # 10× lower than F2/F3
        joint_contrastive_warmup=20,     # ramp λ over 20 epochs after unfreeze
        contrastive_temperature=0.07,
        contrastive_proj_dim=128,
    ),

    # F5: Gradient detach — contrastive only trains projector, not backbone
    504: ExpConfig(
        exp_id=504, name="f5_joint_detach",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        batch_size=32,
        total_epochs=80,
        freeze_epochs=5,
        lr_finetune=1e-4,
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
        use_joint_contrastive=True,
        joint_contrastive_weight=0.1,
        detach_contrastive_backbone=True, # backbone gradient from OrdSupCon blocked
        contrastive_temperature=0.07,
        contrastive_proj_dim=128,
    ),

    # === Phase C: IDRiD Supplement (docs/03-ordinal-supcon.md §11.3) ===

    # C1: IDRiD Grade 3+4 supplement with D1 regularization recipe
    301: ExpConfig(
        exp_id=301, name="c1_idrid_supplement",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        total_epochs=80,
        freeze_epochs=5,
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
        use_idrid_supplement=True,
        idrid_processed_dir=str(IDRID_PROCESSED_DIR),
        idrid_csv=str(IDRID_CSV),
    ),

    # === Phase G: Ordinal-Consistent Fine-tuning (docs/04-coral.md) ===

    # A1-v3: Re-train APTOS backbone with 40 contrastive epochs for Phase G
    605: ExpConfig(
        exp_id=605, name="a1v3_ordsupcon_40ep",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        use_contrastive_pretrain=True,
        contrastive_epochs=40,
        contrastive_lr=1e-3,
        contrastive_temperature=0.07,
        contrastive_proj_dim=128,
        contrastive_data="aptos",
        total_epochs=1,
        freeze_epochs=1,
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # G1: CORN on ImageNet backbone (GATE experiment)
    600: ExpConfig(
        exp_id=600, name="g1_corn_imagenet",
        aug_level=2, loss_type="corn", use_class_weights=False,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        total_epochs=80,
        freeze_epochs=5,
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # G2: CORN on A1-v3 backbone (ordinal pretrain + ordinal finetune)
    601: ExpConfig(
        exp_id=601, name="g2_corn_a1",
        aug_level=2, loss_type="corn", use_class_weights=False,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        total_epochs=80,
        freeze_epochs=5,
        load_backbone=str(
            CHECKPOINT_DIR / "exp605_a1v3_ordsupcon_40ep" / "exp605_a1v3_ordsupcon_40ep_backbone.pth"
        ),
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # G3: CORN on A2 backbone (test if CORN fixes domain gap)
    602: ExpConfig(
        exp_id=602, name="g3_corn_a2",
        aug_level=2, loss_type="corn", use_class_weights=False,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        total_epochs=80,
        freeze_epochs=5,
        load_backbone=str(
            CHECKPOINT_DIR / "exp200_a2_ordsupcon_eyepacs" / "exp200_a2_ordsupcon_eyepacs_backbone.pth"
        ),
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # G4: EMD loss on A1-v3 backbone (ordinal loss, no architecture change)
    603: ExpConfig(
        exp_id=603, name="g4_emd_a1",
        aug_level=2, loss_type="emd", use_class_weights=False,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        total_epochs=80,
        freeze_epochs=5,
        load_backbone=str(
            CHECKPOINT_DIR / "exp605_a1v3_ordsupcon_40ep" / "exp605_a1v3_ordsupcon_40ep_backbone.pth"
        ),
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # G5: Cumulative Link BCE on A1-v3 backbone
    604: ExpConfig(
        exp_id=604, name="g5_cumlink_a1",
        aug_level=2, loss_type="cumlink", use_class_weights=False,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        total_epochs=80,
        freeze_epochs=5,
        load_backbone=str(
            CHECKPOINT_DIR / "exp605_a1v3_ordsupcon_40ep" / "exp605_a1v3_ordsupcon_40ep_backbone.pth"
        ),
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # === Phase H: LP-FT for OrdSupCon (docs/05-phase-h.md) ===

    # H0: Pure linear probe on A2 backbone (diagnostic)
    700: ExpConfig(
        exp_id=700, name="h0_linear_probe_a2",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        total_epochs=50,
        freeze_epochs=50,  # never unfreezes — pure linear probe
        load_backbone=str(
            CHECKPOINT_DIR / "exp200_a2_ordsupcon_eyepacs" / "exp200_a2_ordsupcon_eyepacs_backbone.pth"
        ),
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # H1: A2 backbone + D1 recipe (fair comparison — the missing experiment)
    701: ExpConfig(
        exp_id=701, name="h1_ordsupcon_d1recipe",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        total_epochs=80,
        freeze_epochs=5,
        lr_finetune=1e-4,
        load_backbone=str(
            CHECKPOINT_DIR / "exp200_a2_ordsupcon_eyepacs" / "exp200_a2_ordsupcon_eyepacs_backbone.pth"
        ),
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # H2: A2 backbone + LP-FT (core experiment)
    702: ExpConfig(
        exp_id=702, name="h2_lpft_a2",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        total_epochs=80,
        freeze_epochs=20,       # LP-FT: long linear probe
        lr_finetune=1e-5,       # LP-FT: 10x gentler backbone fine-tuning
        load_backbone=str(
            CHECKPOINT_DIR / "exp200_a2_ordsupcon_eyepacs" / "exp200_a2_ordsupcon_eyepacs_backbone.pth"
        ),
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # H3: A2 backbone + LP-FT + layer-wise LR decay
    703: ExpConfig(
        exp_id=703, name="h3_lpft_layerwise_a2",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        total_epochs=80,
        freeze_epochs=20,           # LP-FT long linear probe
        lr_finetune=1e-4,           # base LR for head (layer1 gets decay^4 x this)
        layerwise_lr_decay=0.316,   # geometric decay per layer group
        load_backbone=str(
            CHECKPOINT_DIR / "exp200_a2_ordsupcon_eyepacs" / "exp200_a2_ordsupcon_eyepacs_backbone.pth"
        ),
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),

    # H4: A1-v3 backbone + LP-FT (same-domain OrdSupCon)
    704: ExpConfig(
        exp_id=704, name="h4_lpft_a1",
        aug_level=2, loss_type="focal", use_class_weights=True,
        use_gem=True,
        head_dropout=0.3,
        weight_decay=1e-4,
        scheduler="cosine_decay",
        total_epochs=80,
        freeze_epochs=20,
        lr_finetune=1e-5,
        load_backbone=str(
            CHECKPOINT_DIR / "exp605_a1v3_ordsupcon_40ep" / "exp605_a1v3_ordsupcon_40ep_backbone.pth"
        ),
        oversample_target=1000,
        oversample_dir=str(ROOT_DIR / "data" / "train_oversampled"),
    ),
}


def get_config(exp_id: int) -> ExpConfig:
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment id: {exp_id}. Valid: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[exp_id]
