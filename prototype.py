import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================
# 1. COMPLEX LINEAR LAYER
# ==============================================================

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.Wr = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.Wi = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

        self.br = nn.Parameter(torch.zeros(out_features))
        self.bi = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        xr, xi = x.real, x.imag

        real = F.linear(xr, self.Wr, self.br) - F.linear(xi, self.Wi)
        imag = F.linear(xr, self.Wi, self.bi) + F.linear(xi, self.Wr)

        return torch.complex(real, imag)


# ==============================================================
# 2. COMPLEX DROPOUT
# ==============================================================

class ComplexDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand_like(x.real) > self.p).float() / (1 - self.p)
        return torch.complex(x.real * mask, x.imag * mask)


# ==============================================================
# 3. COMPLEX FOURIER ATTENTION
# ==============================================================

class QuantumFourierAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads
        self.scale = 1 / math.sqrt(self.dh)

        self.to_q = ComplexLinear(dim, dim)
        self.to_k = ComplexLinear(dim, dim)
        self.to_v = ComplexLinear(dim, dim)
        self.to_out = ComplexLinear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape

        Q = self.to_q(x).view(B, N, self.heads, self.dh).transpose(1, 2)
        K = self.to_k(x).view(B, N, self.heads, self.dh).transpose(1, 2)
        V = self.to_v(x).view(B, N, self.heads, self.dh).transpose(1, 2)

        # FFT in sequence axis
        Qf = torch.fft.fft(Q, dim=2)
        Kf = torch.fft.fft(K, dim=2)
        Vf = torch.fft.fft(V, dim=2)

        # Complex attention scores
        scores = torch.einsum("bhid,bhjd->bhij", Qf, Kf.conj()) * self.scale

        # Softmax only on real part
        attn = F.softmax(scores.real, dim=-1)
        attn = self.attn_dropout(attn)
        attn = attn.to(torch.complex64)

        # Apply attention to V
        out_f = torch.einsum("bhij,bhjd->bhid", attn, Vf)

        # Back to time domain
        out = torch.fft.ifft(out_f, dim=2)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.to_out(out)


# ==============================================================
# 4. COMPLEX FEEDFORWARD
# ==============================================================

class QuantumFeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        hidden = dim * mult
        self.lin1 = ComplexLinear(dim, hidden)
        self.lin2 = ComplexLinear(hidden, dim)
        self.drop = ComplexDropout(dropout)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.complex(F.gelu(x.real), F.gelu(x.imag))
        x = self.drop(x)
        x = self.lin2(x)
        return self.drop(x)


# ==============================================================
# 5. COMPLEX SCHRÖDINGER EVOLUTION
# ==============================================================

class SchrodingerEvolution(nn.Module):
    def __init__(self, dim, dt=0.05):
        super().__init__()
        H = torch.randn(dim, dim) * 0.02
        H = (H + H.T) / 2  # Hermitian
        self.H = nn.Parameter(H)
        self.dt = nn.Parameter(torch.tensor(dt))

    def forward(self, x):
        B, N, D = x.shape

        x_flat = x.reshape(B*N, D)
        U = torch.matrix_exp(-1j * self.H * self.dt)  # unitary evolution
        out = torch.matmul(x_flat, U)
        return out.reshape(B, N, D)


# ==============================================================
# 6. FULL COMPLEX BLOCK
# ==============================================================

class WaveTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.attn = QuantumFourierAttention(dim, heads, dropout)
        self.ff = QuantumFeedForward(dim, 4, dropout)
        self.sch = SchrodingerEvolution(dim)
        self.drop = ComplexDropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(x))
        x = x + self.drop(self.ff(x))
        x = self.sch(x)
        return x


# ==============================================================
# 7. FULL COMPLEX QUANTUM-WAVE TRANSFORMER
# ==============================================================

class QuantumWaveTransformer(nn.Module):
    def __init__(self, dim=64, depth=4, heads=8):
        super().__init__()
        self.input_proj = ComplexLinear(dim, dim)
        self.blocks = nn.ModuleList([WaveTransformerBlock(dim, heads) for _ in range(depth)])
        self.output_proj = nn.Linear(dim*2, dim)

    def forward(self, x):
        x = torch.complex(x, torch.zeros_like(x))
        x = self.input_proj(x)

        for blk in self.blocks:
            x = blk(x)

        x_real = torch.cat([x.real, x.imag], dim=-1)
        return self.output_proj(x_real)


# ==============================================================
# 8. PHYSICS DATA GENERATORS
# ==============================================================

def generate_rotation(batch, length=64):
    t = torch.linspace(0, 2*math.pi, length)
    x = torch.sin(t); y = torch.cos(t)
    base = torch.stack([x, y], dim=-1)
    base = base.unsqueeze(0).repeat(batch, 1, 1)
    return torch.cat([base, torch.zeros(batch, length, 62)], dim=-1)

def generate_RLC(batch, length=64):
    t = torch.linspace(0, 8, length)
    alpha = 0.3; omega = 1.5
    I = torch.exp(-alpha*t)*torch.cos(omega*t)
    I = I / (I.abs().max()+1e-6)
    base = torch.stack([I, torch.gradient(I)[0]], dim=-1)
    base = base.unsqueeze(0).repeat(batch, 1, 1)
    return torch.cat([base, torch.zeros(batch, length, 62)], dim=-1)


# ==============================================================
# 9. TRAINING
# ==============================================================

def train(model, generator, steps=800):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step in range(steps):
        x = generator(8).float()
        pred = model(x)
        loss = F.mse_loss(pred, x)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"[{step}] Loss = {loss.item():.6f}")

    return model


# ==============================================================
# 10. VISUALIZATION
# ==============================================================

def visualize(model, generator):
    x = generator(1).float()
    pred = model(x).detach()

    gt = x[0, :, 0].numpy()
    pr = pred[0, :, 0].numpy()

    plt.plot(gt, label="GT"); plt.plot(pr, label="Pred")
    plt.legend(); plt.show()

# ==============================================================
# 11. SCHRÖDINGER WAVEFUNCTION DATA (Quantum Harmonic Oscillator)
# ==============================================================

def schrodinger_dataset(batch=4, T=64, N=256, dt=0.01):
    """
    Generates ψ(x,t) using split-step Fourier Schrödinger solver.
    Returns complex tensor: (batch, T, N)
    """
    x = np.linspace(-5, 5, N)
    dx = x[1] - x[0]

    # momentum space
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    # Harmonic oscillator potential V(x) = 1/2 x^2
    V = 0.5 * x**2

    all_sequences = []

    for _ in range(batch):
        # initial wave packet
        psi = np.exp(-x**2) * np.exp(1j * np.random.uniform(1, 4) * x)
        psi /= np.sqrt(np.sum(np.abs(psi)**2))

        seq = []

        for _ in range(T):
            seq.append(psi.copy())

            # --- Split-step evolution ---
            psi *= np.exp(-1j * V * dt / 2)
            psi_k = np.fft.fft(psi)
            psi_k *= np.exp(-1j * (k**2)/2 * dt)
            psi = np.fft.ifft(psi_k)
            psi *= np.exp(-1j * V * dt / 2)

            # normalize
            psi /= np.sqrt(np.sum(np.abs(psi)**2))

        all_sequences.append(np.stack(seq))

    return torch.from_numpy(np.stack(all_sequences)).to(torch.complex64)


# ==============================================================
# 12. PREPARE DATA FOR TRANSFORMER (Reduce N → 64 dims using FFT)
# ==============================================================

def prepare_schrodinger_batch(batch=4):
    psi = schrodinger_dataset(batch)  # (B, T, N=256)

    # compress wavefunction → first 32 complex Fourier modes → 64 real dims
    psi_fft = torch.fft.fft(psi, dim=-1)
    psi_modes = psi_fft[..., :32]  # (B, T, 32)

    # convert to real (transformer expects real input)
    psi_real = torch.cat([psi_modes.real, psi_modes.imag], dim=-1)  # (B,T,64)

    return psi_real.float(), psi_real.float()


# ==============================================================
# 13. TRAINING FOR SCHRÖDINGER EQUATION
# ==============================================================

def train_schrodinger(model, steps=1500):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step in range(steps):
        x, target = prepare_schrodinger_batch(4)
        pred = model(x)

        loss = F.mse_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"[Schr {step}] loss = {loss.item():.6f}")


# ==============================================================
# 14. VISUALIZATION OF |ψ|²
# ==============================================================

def visualize_schrodinger(model):
    x, _ = prepare_schrodinger_batch(1)
    pred = model(x).detach()  # (1,T,64)

    # reconstruct complex 32-mode wavefunction
    real = pred[0, :, :32]
    imag = pred[0, :, 32:]
    psi_fft = torch.complex(real, imag)

    # inverse FFT back to 256 points
    psi = torch.fft.ifft(psi_fft, n=256, dim=-1)
    prob = psi.abs().cpu().numpy()**2  # |ψ|²

    plt.imshow(prob, aspect='auto', cmap='plasma')
    plt.title("|ψ(x,t)|² - Schrödinger Prediction")
    plt.xlabel("x")
    plt.ylabel("time")
    plt.colorbar()
    plt.show()

# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=== Complex Quantum Wave Transformer ===")

    model = QuantumWaveTransformer()

    print("\nTraining on Rotation...")
    train(model, generate_rotation)
    visualize(model, generate_rotation)

    print("\nTraining on EM RLC Circuit...")
    train(model, generate_RLC)
    visualize(model, generate_RLC)

    print("\nTraining on Schrödinger Wave Equation...")
    train_schrodinger(model)
    visualize_schrodinger(model)

if __name__ == "__main__":
    main()