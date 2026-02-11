
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

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

    def train_adam(self, iters=4000, lr=1e-3, print_every=200):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for it in range(1, iters + 1):
            opt.zero_grad()
            L_total, L_pde, L_ic, L_bc = self.loss()
            L_total.backward()
            opt.step()
            self._log(L_total, L_pde, L_ic, L_bc)
            if it % print_every == 0:
                pass # Suppressed per-step printing for cleaner multi-run output

    def train_lbfgs(self, max_iter=400, print_every=50):
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
#GER method
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

        G = self.gradient_features(self.X_g)
        H, logdet = self.gaussian_entropy(G)

        L_total = (
            self.lambda_pde * L_pde +
            self.lambda_bc * L_bc +
            self.lambda_ic * L_ic -
            self.lambda_h * H
        )

        return L_total, L_pde, L_ic, L_bc, H, logdet

    def train_adam(self, iters=4000, lr=1e-3, print_every=200):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for it in range(1, iters + 1):
            opt.zero_grad()
            L_total, L_pde, L_ic, L_bc, H, logdet = self.loss()
            L_total.backward()
            opt.step()
            self._log(L_total, L_pde, L_ic, L_bc, H, logdet)
            if it % print_every == 0:
                pass

    def train_lbfgs(self, max_iter=400, print_every=50):
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

def plot_loss_curves(loss_hist: dict, title_prefix: str):
    plt.figure(figsize=(8, 5))
    plt.semilogy(loss_hist["total"], label="total")
    plt.semilogy(loss_hist["pde"], label="PDE")
    plt.semilogy(loss_hist["bc"], label="BC")
    plt.semilogy(loss_hist["ic"], label="IC (0)")
    plt.xlabel("Loss evaluation")
    plt.ylabel("Loss (log scale)")
    plt.title(f"{title_prefix} Loss Curves")
    plt.legend()
    plt.tight_layout()

def make_shared_training_points(device, seed, N_f=20000, N_bc=2000):
    # Using specific seed for data generation
    torch.manual_seed(seed)
    
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

def make_entropy_points(device, N_g=2048, seed=999):
    torch.manual_seed(seed)
    x_g = -1 + 2 * torch.rand(N_g, 1, device=device)
    y_g = -1 + 2 * torch.rand(N_g, 1, device=device)
    X_g = torch.cat([x_g, y_g], dim=1)
    X_g.requires_grad_(True)
    return X_g


def run_trial(model_type, trial_seed, device, params):
    # use trial_seed for data generation so points are different per run
    shared = make_shared_training_points(device=device, seed=trial_seed, N_f=params['N_f'], N_bc=params['N_bc'])
    
    # init model
    if model_type == 'PINN':
        model = PINN(
            hidden=params['hidden'], depth=params['depth'], act=params['act'],
            lambda_pde=params['lambda_pde'], lambda_ic=params['lambda_ic'], lambda_bc=params['lambda_bc'],
            seed=trial_seed, device=device
        )
    else:
        model = GERPINN(
            hidden=params['hidden'], depth=params['depth'], act=params['act'],
            lambda_pde=params['lambda_pde'], lambda_ic=params['lambda_ic'], lambda_bc=params['lambda_bc'],
            lambda_h=params['lambda_h'], eps_cov=params['eps_cov'], 
            use_extended_grad=True, seed=trial_seed, device=device
        )
        # Assign entropy pts for GERPINN
        #  vary seed for entropy pts
        model.X_g = make_entropy_points(device=device, N_g=params['N_g'], seed=trial_seed + 9999)

    assign_shared_points(model, shared)

    # Adam
    model.train_adam(iters=params['adam_iters'], lr=params['adam_lr'], print_every=params['print_every'])
    
    # LBFGS
    if model_type == 'GERPINN':
        #entropy term may interfere with BFGS
        model.lambda_h = 0.0
        
    model.train_lbfgs(max_iter=params['lbfgs_iters'], print_every=params['print_every'])

    # 4. Evaluation
    Nx, Ny = 201, 201
    x_grid, y_grid, U_pred = model.predict_on_grid(Nx=Nx, Ny=Ny)
    _, _, U_true = analytic_on_grid(Nx=Nx, Ny=Ny, device=device)
    
    error = avg_l2_error(U_pred, U_true)
    
    return error, model, U_pred, U_true, x_grid, y_grid

def main():
    # Hyperparameters
    params = {
        'hidden': 64, 'depth': 4, 'act': nn.Tanh,
        'N_f': 20000, 'N_bc': 2000, 'N_g': 2048,
        'adam_iters': 4000, 'adam_lr': 1e-3, 'lbfgs_iters': 400,
        'lambda_pde': 1.0, 'lambda_bc': 1.0, 'lambda_ic': 0.0,
        'lambda_h': 1e-3, 'eps_cov': 1e-6,
        'print_every': 1000 
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    pinn_errors = []
    gerpinn_errors = []
    
    num_runs = 10
    
    # hold the last runs data for plotting
    last_pinn_data = None
    last_ger_data = None

    print(f"\nStarting {num_runs} independent runs for PINN and GERPINN...")
    print("-" * 60)
    print(f"{'Run':<5} | {'PINN Error':<15} | {'GERPINN Error':<15}")
    print("-" * 60)

    start_time = time.time()

    for i in range(num_runs):
        # Use different seeds for PINN and GERPINN to ensure independence
        # Ensure they are deterministic per run index
        seed_pinn = i
        seed_ger = i + 1000 
        
        # Train PINN 
        err_pinn, model_pinn, U_pinn, U_true, x, y = run_trial('PINN', seed_pinn, device, params)
        pinn_errors.append(err_pinn)
        
        # Train GERPINN 
        err_ger, model_ger, U_ger, _, _, _ = run_trial('GERPINN', seed_ger, device, params)
        gerpinn_errors.append(err_ger)
        
        print(f"{i+1:<5} | {err_pinn:.5e}     | {err_ger:.5e}")
        
        if i == num_runs - 1:
            last_pinn_data = (model_pinn, U_pinn)
            last_ger_data = (model_ger, U_ger)

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Total time: {total_time:.2f}s")
    
    # Report Statistics
    print("\nFinal Results:")
    print(f"PINN Average L2 Error:    {np.mean(pinn_errors):.6e} +/- {np.std(pinn_errors):.6e}")
    print(f"GERPINN Average L2 Error: {np.mean(gerpinn_errors):.6e} +/- {np.std(gerpinn_errors):.6e}")

    #Evaluations
    (pinn_model, U_van) = last_pinn_data
    (ger_model, U_ger) = last_ger_data
    
    U_all = np.concatenate([U_true.ravel(), U_van.ravel(), U_ger.ravel()])
    umin, umax = float(U_all.min()), float(U_all.max())

    E_van = np.abs(U_van - U_true)
    E_ger = np.abs(U_ger - U_true)
    emax = float(max(E_van.max(), E_ger.max()))

    plot_heatmap(U_true, x, y, "Analytical Solution", vmin=umin, vmax=umax)
    plot_heatmap(U_van, x, y, "Vanilla-PINN Solution", vmin=umin, vmax=umax)
    plot_heatmap(U_ger, x, y, "GERPINN Solution", vmin=umin, vmax=umax)

    plot_heatmap(E_van, x, y, "|Vanilla-PINN - Analytical|", cbar_label="abs error", vmin=0.0, vmax=emax)
    plot_heatmap(E_ger, x, y, "|GERPINN - Analytical|", cbar_label="abs error", vmin=0.0, vmax=emax)

    plot_loss_curves(pinn_model.loss_hist, "PINN")
    plot_loss_curves(ger_model.loss_hist, "GERPINN")

    plt.show()

if __name__ == "__main__":
    main()
