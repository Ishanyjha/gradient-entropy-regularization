import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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

def u_analytic(X: torch.Tensor) -> torch.Tensor:
    x = X[:, 0:1]
    y = X[:, 1:2]
    return torch.exp(x * y)

def poisson_rhs(X: torch.Tensor) -> torch.Tensor:
    x = X[:, 0:1]
    y = X[:, 1:2]
    return torch.exp(x * y) * (x**2 + y**2)

class PINN:
    def __init__(
        self,
        hidden=64,
        depth=4,
        act=nn.Tanh,
        lambda_pde=1.0,
        lambda_ic=0.0,
        lambda_bc=1.0,
        seed=0,
        device=None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(in_dim=2, out_dim=1, hidden=hidden, depth=depth, act=act).to(self.device)
        self.lambda_pde = float(lambda_pde)
        self.lambda_ic = float(lambda_ic)
        self.lambda_bc = float(lambda_bc)
        self.loss_hist = {"total": [], "pde": [], "ic": [], "bc": []}

    def pde_residual(self, X):
        u = self.model(X)
        grads = torch.autograd.grad(
            outputs=u, inputs=X, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )[0]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=X, grad_outputs=torch.ones_like(u_x),
            retain_graph=True, create_graph=True
        )[0][:, 0:1]
        u_yy = torch.autograd.grad(
            outputs=u_y, inputs=X, grad_outputs=torch.ones_like(u_y),
            retain_graph=True, create_graph=True
        )[0][:, 1:2]
        return (u_xx + u_yy) - poisson_rhs(X)

    def loss(self):
        u_left = self.model(self.X_bc_left)
        u_right = self.model(self.X_bc_right)
        u_bottom = self.model(self.X_bc_bottom)
        u_top = self.model(self.X_bc_top)

        L_bc = (
            torch.mean((u_left - self.y_bc_left) ** 2) +
            torch.mean((u_right - self.y_bc_right) ** 2) +
            torch.mean((u_bottom - self.y_bc_bottom) ** 2) +
            torch.mean((u_top - self.y_bc_top) ** 2)
        )

        r = self.pde_residual(self.X_f)
        L_pde = torch.mean(r ** 2)

        L_ic = torch.zeros((), device=self.device)
        L_total = self.lambda_pde * L_pde + self.lambda_bc * L_bc + self.lambda_ic * L_ic
        return L_total, L_pde, L_ic, L_bc

    def train_adam(self, iters=4000, lr=1e-3, print_every=1000):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for it in range(1, iters + 1):
            opt.zero_grad()
            L_total, L_pde, L_ic, L_bc = self.loss()
            L_total.backward()
            opt.step()
            self._log(L_total, L_pde, L_ic, L_bc)
            if it % print_every == 0:
                print(
                    f"  [PINN Adam] iter={it:6d} "
                    f"total={L_total.item():.3e} "
                    f"pde={L_pde.item():.3e} bc={L_bc.item():.3e}"
                )

    def train_lbfgs(self, max_iter=400, print_every=100):
        opt = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=max_iter,
            max_eval=max_iter,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
        )
        self.model.train()
        k = {"i": 0}

        def closure():
            opt.zero_grad()
            L_total, L_pde, L_ic, L_bc = self.loss()
            L_total.backward()
            self._log(L_total, L_pde, L_ic, L_bc)
            k["i"] += 1
            if k["i"] % print_every == 0:
                print(
                    f"  [PINN LBFGS] iter={k['i']:6d} "
                    f"total={L_total.item():.3e}"
                )
            return L_total

        opt.step(closure)

    def _log(self, L_total, L_pde, L_ic, L_bc):
        self.loss_hist["total"].append(float(L_total.item()))
        self.loss_hist["pde"].append(float(L_pde.item()))
        self.loss_hist["ic"].append(float(L_ic.item()))
        self.loss_hist["bc"].append(float(L_bc.item()))

    @torch.no_grad()
    def predict_on_grid(self, Nx=201, Ny=201):
        x = torch.linspace(-1, 1, Nx)
        y = torch.linspace(-1, 1, Ny)
        Xg, Yg = torch.meshgrid(x, y, indexing="ij")
        XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1).to(self.device)
        U = self.model(XY).reshape(Nx, Ny).cpu().numpy()
        return x.cpu().numpy(), y.cpu().numpy(), U

class GERPINN:
    def __init__(
        self,
        hidden=64,
        depth=4,
        act=nn.Tanh,
        lambda_pde=1.0,
        lambda_ic=0.0,
        lambda_bc=1.0,
        lambda_h=1e-3,
        eps_cov=1e-6,
        use_extended_grad=True,
        seed=0,
        device=None,
    ):
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
        self.loss_hist = {
            "total": [], "pde": [], "ic": [], "bc": [],
            "entropy": [], "entropy_term": [], "logdet": []
        }

    def pde_residual(self, X):
        u = self.model(X)
        grads = torch.autograd.grad(
            outputs=u, inputs=X, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )[0]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=X, grad_outputs=torch.ones_like(u_x),
            retain_graph=True, create_graph=True
        )[0][:, 0:1]
        u_yy = torch.autograd.grad(
            outputs=u_y, inputs=X, grad_outputs=torch.ones_like(u_y),
            retain_graph=True, create_graph=True
        )[0][:, 1:2]
        return (u_xx + u_yy) - poisson_rhs(X)

    def gradient_features(self, X):
        u = self.model(X)
        grads = torch.autograd.grad(
            outputs=u, inputs=X, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )[0]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        if self.use_extended_grad:
            return torch.cat([u_x, u_y], dim=1)
        return u_x

    def cov_logdet(self, G):
        N, d = G.shape
        mu = G.mean(dim=0, keepdim=True)
        Gc = G - mu
        Sigma = (Gc.T @ Gc) / (N - 1)
        Sigma = Sigma + self.eps_cov * torch.eye(d, device=G.device, dtype=G.dtype)
        L = torch.linalg.cholesky(Sigma)
        logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
        return Sigma, logdet

    def gaussian_entropy(self, G):
        _, logdet = self.cov_logdet(G)
        d = G.shape[1]
        H = 0.5 * (d * math.log(2.0 * math.pi * math.e) + logdet)
        return H, logdet

    def loss(self):
        u_left = self.model(self.X_bc_left)
        u_right = self.model(self.X_bc_right)
        u_bottom = self.model(self.X_bc_bottom)
        u_top = self.model(self.X_bc_top)

        L_bc = (
            torch.mean((u_left - self.y_bc_left) ** 2) +
            torch.mean((u_right - self.y_bc_right) ** 2) +
            torch.mean((u_bottom - self.y_bc_bottom) ** 2) +
            torch.mean((u_top - self.y_bc_top) ** 2)
        )

        r = self.pde_residual(self.X_f)
        L_pde = torch.mean(r ** 2)

        L_ic = torch.zeros((), device=self.device)

        # Uses the same X_g (which we will map to X_f)
        G = self.gradient_features(self.X_g)
        H, logdet = self.gaussian_entropy(G)

        L_total = (
            self.lambda_pde * L_pde +
            self.lambda_bc * L_bc +
            self.lambda_ic * L_ic -
            self.lambda_h * H
        )

        return L_total, L_pde, L_ic, L_bc, H, logdet

    def train_adam(self, iters=4000, lr=1e-3, print_every=1000):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for it in range(1, iters + 1):
            opt.zero_grad()
            L_total, L_pde, L_ic, L_bc, H, logdet = self.loss()
            L_total.backward()
            opt.step()
            self._log(L_total, L_pde, L_ic, L_bc, H, logdet)
            if it % print_every == 0:
                print(
                    f"  [GERPINN Adam] iter={it:6d} "
                    f"total={L_total.item():.3e} "
                    f"pde={L_pde.item():.3e} H={H.item():.3e}"
                )

    def train_lbfgs(self, max_iter=400, print_every=100):
        opt = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=max_iter,
            max_eval=max_iter,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
        )
        self.model.train()
        k = {"i": 0}

        def closure():
            opt.zero_grad()
            L_total, L_pde, L_ic, L_bc, H, logdet = self.loss()
            L_total.backward()
            self._log(L_total, L_pde, L_ic, L_bc, H, logdet)
            k["i"] += 1
            if k["i"] % print_every == 0:
                print(
                    f"  [GERPINN LBFGS] iter={k['i']:6d} "
                    f"total={L_total.item():.3e}"
                )
            return L_total

        opt.step(closure)

    def _log(self, L_total, L_pde, L_ic, L_bc, H, logdet):
        self.loss_hist["total"].append(float(L_total.item()))
        self.loss_hist["pde"].append(float(L_pde.item()))
        self.loss_hist["ic"].append(float(L_ic.item()))
        self.loss_hist["bc"].append(float(L_bc.item()))
        self.loss_hist["entropy"].append(float(H.item()))
        self.loss_hist["entropy_term"].append(float((-self.lambda_h * H).item()))
        self.loss_hist["logdet"].append(float(logdet.item()))

    @torch.no_grad()
    def predict_on_grid(self, Nx=201, Ny=201):
        x = torch.linspace(-1, 1, Nx)
        y = torch.linspace(-1, 1, Ny)
        Xg, Yg = torch.meshgrid(x, y, indexing="ij")
        XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1).to(self.device)
        U = self.model(XY).reshape(Nx, Ny).cpu().numpy()
        return x.cpu().numpy(), y.cpu().numpy(), U

@torch.no_grad()
def analytic_on_grid(Nx=201, Ny=201, device=None):
    device = device or torch.device("cpu")
    x = torch.linspace(-1, 1, Nx)
    y = torch.linspace(-1, 1, Ny)
    Xg, Yg = torch.meshgrid(x, y, indexing="ij")
    XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1).to(device)
    U = u_analytic(XY).reshape(Nx, Ny).cpu().numpy()
    return x.cpu().numpy(), y.cpu().numpy(), U

def avg_l2_error(U_pred: np.ndarray, U_true: np.ndarray) -> float:
    diff = U_pred - U_true
    return float(np.sqrt(np.mean(diff**2)))

def plot_heatmap(U, x, y, title, cbar_label="u(x,y)", vmin=None, vmax=None):
    plt.figure(figsize=(7.5, 6))
    plt.imshow(
        U,
        origin="lower",
        aspect="auto",
        extent=[y.min(), y.max(), x.min(), x.max()],
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label=cbar_label)
    plt.xlabel("y")
    plt.ylabel("x")
    plt.title(title)
    plt.tight_layout()

def plot_avg_loss_curves(loss_hist_list: list, title_prefix: str):
    # Determine max length
    max_len = max(len(h["total"]) for h in loss_hist_list)
    
    # Helper to aggregate a key
    def get_matrix(key):
        arrs = []
        for h in loss_hist_list:
            vals = h[key]
            # Pad with nan or last value. Here we pad with last value for visual continuity
            pad = [vals[-1]] * (max_len - len(vals))
            arrs.append(vals + pad)
        return np.array(arrs) # Shape (N_runs, max_len)

    keys = ["total", "pde", "bc"]
    means = {}
    stds = {}
    
    for k in keys:
        mat = get_matrix(k)
        means[k] = np.mean(mat, axis=0)
        stds[k] = np.std(mat, axis=0)

    plt.figure(figsize=(8, 5))
    for k in keys:
        plt.semilogy(means[k], label=f"{k} (mean)")
        # Fill between std dev
        lower = np.maximum(means[k] - stds[k], 1e-9) # ensure positive for log plot
        upper = means[k] + stds[k]
        plt.fill_between(range(max_len), lower, upper, alpha=0.2)
        
    plt.xlabel("Loss evaluation")
    plt.ylabel("Loss (log scale)")
    plt.title(f"{title_prefix} Avg Loss Curves (10 runs)")
    plt.legend()
    plt.tight_layout()

def make_shared_training_points(device, N_f=120, N_bc=120):
    x_f = -1 + 2 * torch.rand(N_f, 1, device=device)
    y_f = -1 + 2 * torch.rand(N_f, 1, device=device)
    X_f = torch.cat([x_f, y_f], dim=1)
    X_f.requires_grad_(True)

    n_edge = max(1, N_bc // 4)

    y_left = -1 + 2 * torch.rand(n_edge, 1, device=device)
    x_left = -torch.ones(n_edge, 1, device=device)
    X_bc_left = torch.cat([x_left, y_left], dim=1)

    y_right = -1 + 2 * torch.rand(n_edge, 1, device=device)
    x_right = torch.ones(n_edge, 1, device=device)
    X_bc_right = torch.cat([x_right, y_right], dim=1)

    x_bottom = -1 + 2 * torch.rand(n_edge, 1, device=device)
    y_bottom = -torch.ones(n_edge, 1, device=device)
    X_bc_bottom = torch.cat([x_bottom, y_bottom], dim=1)

    x_top = -1 + 2 * torch.rand(n_edge, 1, device=device)
    y_top = torch.ones(n_edge, 1, device=device)
    X_bc_top = torch.cat([x_top, y_top], dim=1)

    with torch.no_grad():
        y_bc_left = u_analytic(X_bc_left)
        y_bc_right = u_analytic(X_bc_right)
        y_bc_bottom = u_analytic(X_bc_bottom)
        y_bc_top = u_analytic(X_bc_top)

    return {
        "X_f": X_f,
        "X_bc_left": X_bc_left, "y_bc_left": y_bc_left,
        "X_bc_right": X_bc_right, "y_bc_right": y_bc_right,
        "X_bc_bottom": X_bc_bottom, "y_bc_bottom": y_bc_bottom,
        "X_bc_top": X_bc_top, "y_bc_top": y_bc_top,
        "n_edge": n_edge
    }

def assign_shared_points(obj, shared):
    obj.X_f = shared["X_f"]
    obj.X_bc_left, obj.y_bc_left = shared["X_bc_left"], shared["y_bc_left"]
    obj.X_bc_right, obj.y_bc_right = shared["X_bc_right"], shared["y_bc_right"]
    obj.X_bc_bottom, obj.y_bc_bottom = shared["X_bc_bottom"], shared["y_bc_bottom"]
    obj.X_bc_top, obj.y_bc_top = shared["X_bc_top"], shared["y_bc_top"]

def main():
    hidden = 64
    depth = 4
    act = nn.Tanh

    # MODIFIED: Changed counts to 120 each as requested
    N_f = 120
    N_bc = 120

    adam_iters = 4000
    adam_lr = 1e-3
    lbfgs_iters = 400

    lambda_pde = 1.0
    lambda_bc = 1.0
    lambda_ic = 0.0
    lambda_h = 1e-3
    eps_cov = 1e-6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # Storage for 10 runs
    N_runs = 10
    l2_van_runs = []
    l2_ger_runs = []
    
    hist_van_runs = []
    hist_ger_runs = []
    
    error_field_van_acc = 0 # Accumulator for absolute error fields
    error_field_ger_acc = 0

    Nx, Ny = 201, 201
    x, y, U_true = analytic_on_grid(Nx=Nx, Ny=Ny, device=device)

    print(f"\nStarting {N_runs} runs. Training on {N_f} PDE points + {N_bc} BC points.")

    for run_i in range(N_runs):
        seed = run_i
        print(f"\n--- Run {run_i+1}/{N_runs} (Seed {seed}) ---")
        
        # 1. New data for each run (or just new seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        shared = make_shared_training_points(device=device, N_f=N_f, N_bc=N_bc)
        
        # 2. Initialize Models
        pinn = PINN(
            hidden=hidden, depth=depth, act=act,
            lambda_pde=lambda_pde, lambda_ic=lambda_ic, lambda_bc=lambda_bc,
            seed=seed, device=device
        )
        assign_shared_points(pinn, shared)

        ger = GERPINN(
            hidden=hidden, depth=depth, act=act,
            lambda_pde=lambda_pde, lambda_ic=lambda_ic, lambda_bc=lambda_bc,
            lambda_h=lambda_h, eps_cov=eps_cov, use_extended_grad=True,
            seed=seed, device=device
        )
        assign_shared_points(ger, shared)
        
        # MODIFIED: GERPINN uses the same collocation points (X_f) for entropy
        ger.X_g = shared["X_f"]

        # 3. Train PINN
        print("Training PINN...")
        pinn.train_adam(iters=adam_iters, lr=adam_lr, print_every=2000)
        pinn.train_lbfgs(max_iter=lbfgs_iters, print_every=400)
        
        # 4. Train GERPINN
        print("Training GERPINN...")
        ger.train_adam(iters=adam_iters, lr=adam_lr, print_every=2000)
        ger.lambda_h = 0.0
        ger.train_lbfgs(max_iter=lbfgs_iters, print_every=400)
        
        # 5. Evaluate
        _, _, U_van = pinn.predict_on_grid(Nx=Nx, Ny=Ny)
        _, _, U_ger = ger.predict_on_grid(Nx=Nx, Ny=Ny)
        
        l2_van = avg_l2_error(U_van, U_true)
        l2_ger = avg_l2_error(U_ger, U_true)
        
        l2_van_runs.append(l2_van)
        l2_ger_runs.append(l2_ger)
        
        hist_van_runs.append(pinn.loss_hist)
        hist_ger_runs.append(ger.loss_hist)
        
        error_field_van_acc += np.abs(U_van - U_true)
        error_field_ger_acc += np.abs(U_ger - U_true)
        
        print(f"Run {run_i+1} L2 Error: PINN={l2_van:.4e}, GER={l2_ger:.4e}")

    # Process averaged results
    l2_van_avg = np.mean(l2_van_runs)
    l2_van_std = np.std(l2_van_runs)
    l2_ger_avg = np.mean(l2_ger_runs)
    l2_ger_std = np.std(l2_ger_runs)
    
    print("\n" + "="*30)
    print("FINAL RESULTS (Average over 10 runs)")
    print("="*30)
    print(f"PINN L2 Error:    {l2_van_avg:.6e} +/- {l2_van_std:.6e}")
    print(f"GERPINN L2 Error: {l2_ger_avg:.6e} +/- {l2_ger_std:.6e}")

    # Average absolute error fields
    E_van_avg = error_field_van_acc / N_runs
    E_ger_avg = error_field_ger_acc / N_runs
    emax = float(max(E_van_avg.max(), E_ger_avg.max()))

    # Plotting
    plot_heatmap(U_true, x, y, "Analytical Solution")
    
    plot_heatmap(E_van_avg, x, y, 
                 "Avg |Vanilla-PINN - Analytic| (10 runs)", 
                 cbar_label="avg abs error", vmin=0.0, vmax=emax)
                 
    plot_heatmap(E_ger_avg, x, y, 
                 "Avg |GERPINN - Analytic| (10 runs)", 
                 cbar_label="avg abs error", vmin=0.0, vmax=emax)

    plot_avg_loss_curves(hist_van_runs, "PINN")
    plot_avg_loss_curves(hist_ger_runs, "GERPINN")

    plt.show()

if __name__ == "__main__":
    main()
