import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io

class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hidden=64, depth=4, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), act()]
        for _ in range(depth):
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

def load_ac_data(path='AC.mat'):
    data = scipy.io.loadmat(path)
    t = data['tt'].flatten()[:, None] # (201, 1)
    x = data['x'].flatten()[:, None]  # (512, 1)
    Exact = np.real(data['uu'])       # (512, 201)
    return x, t, Exact

class PINN:
    def __init__(self, hidden=64, depth=4, act=nn.Tanh, lambda_pde=1.0, lambda_ic=1.0, lambda_bc=1.0, seed=0, device=None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(in_dim=2, out_dim=1, hidden=hidden, depth=depth, act=act).to(self.device)
        self.lambda_pde = float(lambda_pde)
        self.lambda_ic = float(lambda_ic)
        self.lambda_bc = float(lambda_bc)
        self.loss_hist = {"total": []}

    def pde_residual(self, X):
        u = self.model(X)
        grads = torch.autograd.grad(outputs=u, inputs=X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x, u_t = grads[:, 0:1], grads[:, 1:2]
        u_xx = torch.autograd.grad(outputs=u_x, inputs=X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0][:, 0:1]
        return (u_t - 0.0001 * u_xx) + 5.0 * u**3 - 5.0 * u

    def loss(self):
        u_left = self.model(self.X_bc_left)
        u_right = self.model(self.X_bc_right)
        g_left = torch.autograd.grad(u_left, self.X_bc_left, torch.ones_like(u_left), True, True)[0][:, 0:1]
        g_right = torch.autograd.grad(u_right, self.X_bc_right, torch.ones_like(u_right), True, True)[0][:, 0:1]

        L_bc = torch.mean((u_left - u_right) ** 2) + torch.mean((g_left - g_right) ** 2)

        u_ic = self.model(self.X_ic)
        L_ic = torch.mean((u_ic - self.y_ic) ** 2)

        r = self.pde_residual(self.X_f)
        L_pde = torch.mean(r ** 2)

        L_total = self.lambda_pde * L_pde + self.lambda_bc * L_bc + self.lambda_ic * L_ic
        return L_total

    def train_adam(self, iters=4000, lr=1e-3):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in range(iters):
            opt.zero_grad()
            L_total = self.loss()
            L_total.backward()
            opt.step()
            self.loss_hist["total"].append(L_total.item())

    def train_lbfgs(self, max_iter=400):
        opt = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=max_iter)
        self.model.train()
        def closure():
            opt.zero_grad()
            L_total = self.loss()
            L_total.backward()
            self.loss_hist["total"].append(L_total.item())
            return L_total
        opt.step(closure)

    @torch.no_grad()
    def predict(self, x_np, t_np):
        Nx, Nt = x_np.shape[0], t_np.shape[0]
        Xg, Tg = np.meshgrid(x_np.flatten(), t_np.flatten(), indexing="ij")
        XT = torch.tensor(np.stack([Xg.reshape(-1), Tg.reshape(-1)], axis=1), dtype=torch.float32, device=self.device)
        return self.model(XT).reshape(Nx, Nt).cpu().numpy()


class GERPINN:
    def __init__(self, hidden=64, depth=4, act=nn.Tanh, lambda_pde=1.0, lambda_ic=1.0, lambda_bc=1.0, 
                 lambda_h=1e-2, eps_cov=1e-6, use_extended_grad=True, n_bins=20, target_eps=100.0, seed=0, device=None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(in_dim=2, out_dim=1, hidden=hidden, depth=depth, act=act).to(self.device)
        self.lambda_pde = float(lambda_pde)
        self.lambda_ic = float(lambda_ic)
        self.lambda_bc = float(lambda_bc)
        self.lambda_h = float(lambda_h)
        self.eps_cov = float(eps_cov)
        self.use_extended_grad = bool(use_extended_grad)
        self.n_bins = n_bins
        self.target_eps = target_eps
        self.current_eps = 10.0 
        self.loss_hist = {"total": []}

    def pde_residual(self, X):
        u = self.model(X)
        grads = torch.autograd.grad(outputs=u, inputs=X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x, u_t = grads[:, 0:1], grads[:, 1:2]
        u_xx = torch.autograd.grad(outputs=u_x, inputs=X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0][:, 0:1]
        return (u_t - 0.0001 * u_xx) + 5.0 * u**3 - 5.0 * u

    def gradient_features(self, X):
        u = self.model(X)
        grads = torch.autograd.grad(outputs=u, inputs=X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x, u_t = grads[:, 0:1], grads[:, 1:2]
        if self.use_extended_grad:
            u_xx = torch.autograd.grad(outputs=u_x, inputs=X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0][:, 0:1]
            features = torch.cat([u_x, u_t, u_xx], dim=1)
        else:
            features = u_x
        return (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-6)

    def gaussian_entropy(self, G):
        N, d = G.shape
        mu = G.mean(dim=0, keepdim=True)
        Gc = G - mu
        Sigma = (Gc.T @ Gc) / (N - 1) + self.eps_cov * torch.eye(d, device=G.device, dtype=G.dtype)
        L = torch.linalg.cholesky(Sigma)
        logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
        H = 0.5 * (d * math.log(2.0 * math.pi * math.e) + logdet)
        return H

    def loss(self):
        u_left = self.model(self.X_bc_left)
        u_right = self.model(self.X_bc_right)
        g_left = torch.autograd.grad(u_left, self.X_bc_left, torch.ones_like(u_left), True, True)[0][:, 0:1]
        g_right = torch.autograd.grad(u_right, self.X_bc_right, torch.ones_like(u_right), True, True)[0][:, 0:1]
        L_bc = torch.mean((u_left - u_right) ** 2) + torch.mean((g_left - g_right) ** 2)

        u_ic = self.model(self.X_ic)
        L_ic = torch.mean((u_ic - self.y_ic) ** 2)

        t_f = self.X_f[:, 1]
        sorted_idx = torch.argsort(t_f)
        X_f_sorted = self.X_f[sorted_idx]
        
        r = self.pde_residual(X_f_sorted)
        r_sq = r ** 2
        
        chunk_size = len(r_sq) // self.n_bins
        L_bins = [torch.mean(r_sq[i * chunk_size : (i + 1) * chunk_size if i < self.n_bins - 1 else len(r_sq)]) for i in range(self.n_bins)]
            
        W = []
        L_ic_val = L_ic.detach()
        L_bins_val = [l.detach() for l in L_bins]
        
        for i in range(self.n_bins):
            preceding_error = L_ic_val + (torch.stack(L_bins_val[:i]).sum() if i > 0 else 0.0)
            W.append(torch.exp(-self.current_eps * preceding_error))
            
        L_pde_causal = torch.stack([W[i] * L_bins[i] for i in range(self.n_bins)]).mean()

        G = self.gradient_features(self.X_g)
        H = self.gaussian_entropy(G)

        L_total = self.lambda_pde * L_pde_causal + self.lambda_bc * L_bc + self.lambda_ic * L_ic - self.lambda_h * H
        return L_total

    def train_adam(self, iters=4000, lr=1e-3):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for it in range(1, iters + 1):
            self.current_eps = 10.0 + (self.target_eps - 10.0) * (it / iters)
            opt.zero_grad()
            L_total = self.loss()
            L_total.backward()
            opt.step()
            self.loss_hist["total"].append(L_total.item())

    def adaptive_sample(self, num_new_points=1000, pool_size=100000):
        x_pool = -1 + 2 * torch.rand(pool_size, 1, device=self.X_f.device)
        t_pool = torch.rand(pool_size, 1, device=self.X_f.device)
        X_pool = torch.cat([x_pool, t_pool], dim=1).requires_grad_(True)
        
        self.model.eval()
        r = self.pde_residual(X_pool)
        self.model.train()
        
        p = (r**2).detach().squeeze()
        p = p / torch.sum(p)
        
        idx = torch.multinomial(p, num_new_points, replacement=False)
        X_new = X_pool[idx].detach().requires_grad_(True)
        
        self.X_f = torch.cat([self.X_f, X_new], dim=0)
        self.X_g = self.X_f 

    def train_lbfgs(self, max_iter=400):
        opt = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=max_iter)
        self.model.train()
        def closure():
            opt.zero_grad()
            L_total = self.loss()
            L_total.backward()
            self.loss_hist["total"].append(L_total.item())
            return L_total
        opt.step(closure)

    @torch.no_grad()
    def predict(self, x_np, t_np):
        Nx, Nt = x_np.shape[0], t_np.shape[0]
        Xg, Tg = np.meshgrid(x_np.flatten(), t_np.flatten(), indexing="ij")
        XT = torch.tensor(np.stack([Xg.reshape(-1), Tg.reshape(-1)], axis=1), dtype=torch.float32, device=self.device)
        return self.model(XT).reshape(Nx, Nt).cpu().numpy()


def avg_l2_error(pred, true):
    return float(np.sqrt(np.mean((pred - true)**2)))

def make_shared_bc_ic(device, x_truth, u_truth_at_t0, N_bc=956, noise_level=0.0):
    n_edge = max(1, N_bc // 2)
    t_bc = torch.rand(n_edge, 1, device=device)
    
    X_bc_left = torch.cat([-torch.ones_like(t_bc), t_bc], dim=1).requires_grad_(True)
    X_bc_right = torch.cat([torch.ones_like(t_bc), t_bc], dim=1).requires_grad_(True)

    idx = np.random.choice(x_truth.shape[0], n_edge, replace=False)
    x_ic_vals = x_truth[idx, :]
    u_ic_vals = u_truth_at_t0[idx, :]

    X_ic = torch.tensor(np.hstack([x_ic_vals, np.zeros_like(x_ic_vals)]), dtype=torch.float32, device=device)
    
    y_ic_clean = torch.tensor(u_ic_vals, dtype=torch.float32, device=device)
    
    # Inject Gaussian noise into the Initial Condition
    if noise_level > 0.0:
        std_ic = torch.std(y_ic_clean)
        noise = torch.randn_like(y_ic_clean) * std_ic * noise_level
        y_ic = y_ic_clean + noise
    else:
        y_ic = y_ic_clean

    return {"X_bc_left": X_bc_left, "X_bc_right": X_bc_right, "X_ic": X_ic, "y_ic": y_ic}

def make_domain_points(device, N_f):
    x_f = -1 + 2 * torch.rand(N_f, 1, device=device)
    t_f = torch.rand(N_f, 1, device=device)
    return torch.cat([x_f, t_f], dim=1).requires_grad_(True)

def assign_bc_ic_points(obj, shared):
    obj.X_bc_left = shared["X_bc_left"]
    obj.X_bc_right = shared["X_bc_right"]
    obj.X_ic = shared["X_ic"]
    obj.y_ic = shared["y_ic"]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    x_truth, t_truth, U_true = load_ac_data('AC.mat')
    u_truth_at_t0 = U_true[:, 0:1]

    # Test conditions
    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.40] # 0% to 40% noise on IC
    N_f_pinn = 5000
    N_f_ger_base = 4000
    N_f_ger_adapt = 1000
    
    adam_iters = 4000
    lbfgs_iters = 400
    
    pinn_errors = []
    ger_errors = []

    print(f"\nTesting Robustness to Initial Condition Noise on Allen-Cahn Equation")
    print(f"PINN: {N_f_pinn} pts | GERPINN: {N_f_ger_base} base + {N_f_ger_adapt} adaptive pts")

    for noise in noise_levels:
        print(f"\n=== Training with {noise*100:.0f}% IC Noise ===")
        
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        shared_bc_ic = make_shared_bc_ic(device, x_truth, u_truth_at_t0, N_bc=956, noise_level=noise)

        # 1. PINN Setup
        pinn = PINN(hidden=64, depth=4, lambda_pde=1.0, lambda_ic=75, lambda_bc=75, seed=seed, device=device)
        assign_bc_ic_points(pinn, shared_bc_ic)
        pinn.X_f = make_domain_points(device, N_f_pinn)

        # 2. GERPINN Setup
        ger = GERPINN(hidden=64, depth=4, lambda_pde=1.0, lambda_ic=75, lambda_bc=75, 
                      lambda_h=1e-2, n_bins=20, target_eps=100.0, seed=seed, device=device)
        assign_bc_ic_points(ger, shared_bc_ic)
        ger.X_f = make_domain_points(device, N_f_ger_base)
        ger.X_g = ger.X_f

        # --- Train PINN ---
        print("  Training PINN...")
        pinn.train_adam(iters=adam_iters)
        pinn.train_lbfgs(max_iter=lbfgs_iters)
        
        # --- Train GERPINN ---
        print("  Training GERPINN (Phase 1)...")
        ger.lambda_h = 1e-2 
        ger.train_adam(iters=adam_iters)
        
        print("  Adaptive Sampling...")
        ger.adaptive_sample(num_new_points=N_f_ger_adapt)
        
        print("  Training GERPINN (Phase 2)...")
        ger.lambda_h = 1e-4 
        ger.train_lbfgs(max_iter=lbfgs_iters)

        # --- Evaluate ---
        u_pinn_pred = pinn.predict(x_truth, t_truth)
        u_ger_pred = ger.predict(x_truth, t_truth)
        
        err_pinn = avg_l2_error(u_pinn_pred, U_true)
        err_ger = avg_l2_error(u_ger_pred, U_true)
        
        pinn_errors.append(err_pinn)
        ger_errors.append(err_ger)
        
        print(f"  -> PINN L2 Error: {err_pinn:.4e}")
        print(f"  -> GERPINN L2 Error: {err_ger:.4e}")

    # Plotting Results
    plt.figure(figsize=(8, 6))
    plt.plot(np.array(noise_levels) * 100, pinn_errors, '-o', label='Vanilla PINN', linewidth=2, markersize=8)
    plt.plot(np.array(noise_levels) * 100, ger_errors, '-s', label='GERPINN', linewidth=2, markersize=8)
    plt.xlabel('Gaussian Noise in Initial Condition (%)', fontsize=12)
    plt.ylabel('Average L2 Error', fontsize=12)
    plt.title('Robustness to Data Noise (Allen-Cahn Equation)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
