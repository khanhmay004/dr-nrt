from src.models import OrdinalPrototypeHead
from src.losses import CLOCLoss, RnCLoss, SORDLoss, LogitAdjustedCE
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# 1. OrdinalPrototypeHead — all 5 prototypes should be distinct unit vectors
head = OrdinalPrototypeHead(feat_dim=64, num_classes=5, scale=20.0)
feat = torch.randn(8, 64)
logits = head(feat)
print('Head:', logits.shape, 'range:', f'{logits.min():.3f}..{logits.max():.3f}')

v1_n = F.normalize(head.v1, dim=0)
v2_perp = head.v2 - (head.v2 @ v1_n) * v1_n
v2_n = F.normalize(v2_perp, dim=0)
ks = torch.arange(5, dtype=torch.float32)
angles = (ks - 2) * head.angular_spacing
mu = torch.cos(angles).unsqueeze(1)*v1_n.unsqueeze(0) + torch.sin(angles).unsqueeze(1)*v2_n.unsqueeze(0)
mu_n = F.normalize(mu, dim=1)
cos_table = mu_n @ mu_n.T
print('Proto norms:', [round(x, 4) for x in mu_n.norm(dim=1).tolist()])
print('Adjacent cos:', [round(cos_table[k, k+1].item(), 4) for k in range(4)])
print('Far cos(0,4):', round(cos_table[0, 4].item(), 4))

# 2. RnCLoss — single-numerator paper form; backward reaches features
rnc = RnCLoss(temperature=2.0)
feat = F.normalize(torch.randn(8, 16), dim=1).requires_grad_(True)
labels = torch.tensor([0, 0, 1, 1, 2, 3, 4, 4])
loss = rnc(feat, labels)
loss.backward()
print(f'RnCLoss: {loss.item():.4f}  grad_norm: {feat.grad.norm().item():.4f}')

# 3. CLOCLoss — softplus keeps cumulative_margin non-decreasing
cloc = CLOCLoss(num_classes=5, temperature=0.07)
feat = F.normalize(torch.randn(10, 16), dim=1).requires_grad_(True)
labels = torch.tensor([0, 0, 1, 2, 2, 3, 3, 4, 4, 4])
loss = cloc(feat, labels)
loss.backward()
print(f'CLOCLoss: {loss.item():.4f}  grad_norm: {feat.grad.norm().item():.4f}')

# 4. SORDLoss + LogitAdjustedCE
sord = SORDLoss(num_classes=5)
logits = torch.randn(8, 5, requires_grad=True)
targets = torch.tensor([0, 1, 2, 3, 4, 2, 3, 1])
print(f'SORDLoss: {sord(logits, targets).item():.4f}')

la_ce = LogitAdjustedCE(class_counts=[1534, 317, 850, 164, 249], tau=1.0)
print(f'LA-CE: {la_ce(logits, targets).item():.4f}')

print('OK')