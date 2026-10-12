"""
TinyInfer step 1: train a small CNN on CIFAR-10.

Run in Colab with a GPU runtime. Takes ~5-8 minutes for 5 epochs, enough to get
a reasonable (not SOTA) baseline — the point of this project is the inference
pipeline, not squeezing out max accuracy.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class TinyMobileNet(nn.Module):
    """A small MobileNet-style CNN: depthwise-separable convs, ~137K params.
    Small enough to train fast, structured enough that quantization / kernel
    fusion behave like a real edge-deployment model rather than a toy."""

    def __init__(self, num_classes=10):
        super().__init__()

        def dw_sep(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            dw_sep(32, 64),
            dw_sep(64, 128, stride=2),
            dw_sep(128, 128),
            dw_sep(128, 256, stride=2),
            dw_sep(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def get_loaders(batch_size=128):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main(epochs=5):
    train_loader, test_loader = get_loaders()
    model = TinyMobileNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
        sched.step()
        train_loss = running_loss / len(train_loader.dataset)
        acc = evaluate(model, test_loader)
        print(f"epoch {epoch+1}/{epochs}  loss={train_loss:.4f}  test_acc={acc:.4f}  time={time.time()-t0:.1f}s")

    torch.save(model.state_dict(), "baseline_fp32.pt")
    print("Saved baseline_fp32.pt")
    print(f"Final test accuracy: {evaluate(model, test_loader):.4f}")


if __name__ == "__main__":
    main()
