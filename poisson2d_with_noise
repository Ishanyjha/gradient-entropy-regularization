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

    def train_adam(self, iters=4000, lr=1e-3, verbose=False):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for it in range(1, iters + 1):
            opt.zero_grad()
            L_total, L_pde, L_ic, L_bc = self.loss()
            L_total.backward()
            opt.step()
            self._log(L_total, L_pde, L_ic, L_bc)
            if verbose and it % 1000 == 0:
                 print(f"    [PINN Adam] iter={it} total={L_total.item():.3e}")

    def train_lbfgs(self, max_iter=400, verbose=False):
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
        def closure():
            opt.zero_grad()
            L_total, L_pde, L_ic, L_bc = self.loss()
            L_total.backward()
            self._log(L_total, L_pde, L_ic, L_bc)
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

        G = self.gradient_features(self.X_g)
        H, logdet = self.gaussian_entropy(G)

        L_total = (
            self.lambda_pde * L_pde +
            self.lambda_bc * L_bc +
            self.lambda_ic * L_ic -
            self.lambda_h * H
        )

        return L_total, L_pde, L_ic, L_bc, H, logdet

    def train_adam(self, iters=4000, lr=1e-3, verbose=False):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for it in range(1, iters + 1):
            opt.zero_grad()
            L_total, L_pde, L_ic, L_bc, H, logdet = self.loss()
            L_total.backward()
            opt.step()
            self._log(L_total, L_pde, L_ic, L_bc, H, logdet)
            if verbose and it % 1000 == 0:
                print(f"    [GERPINN Adam] iter={it} total={L_total.item():.3e}")

    def train_lbfgs(self, max_iter=400, verbose=False):
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
        def closure():
            opt.zero_grad()
            L_total, L_pde, L_ic, L_bc, H, logdet = self.loss()
            L_total.backward()
            self._log(L_total, L_pde, L_ic, L_bc, H, logdet)
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

def make_shared_training_points(device, N_f=500, N_bc=250, noise_level=0.0):
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

        if noise_level > 0.0:
            all_y = torch.cat([y_bc_left, y_bc_right, y_bc_bottom, y_bc_top])
            std_y = torch.std(all_y)
            
            def add_noise(y):
                return y + noise_level * std_y * torch.randn_like(y)

            y_bc_left = add_noise(y_bc_left)
            y_bc_right = add_noise(y_bc_right)
            y_bc_bottom = add_noise(y_bc_bottom)
            y_bc_top = add_noise(y_bc_top)

    return {
        "X_f": X_f,
        "X_bc_left": X_bc_left, "y_bc_left": y_bc_left,
        "X_bc_right": X_bc_right, "y_bc_right": y_bc_right,
        "X_bc_bottom": X_bc_bottom, "y_bc_bottom": y_bc_bottom,
        "X_bc_top": X_bc_top, "y_bc_top": y_bc_top,
        "n_edge": n_edge
    }
def make_entropy_points(device, N_g=100, seed=123):
    torch.manual_seed(seed)
    x_g = -1 + 2 * torch.rand(N_g, 1, device=device)
    y_g = -1 + 2 * torch.rand(N_g, 1, device=device)
    X_g = torch.cat([x_g, y_g], dim=1)
    X_g.requires_grad_(True)
    return X_g
    
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
    
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25] 
    num_trials = 8          
    
    N_f = 500   
    N_bc = 250  

    adam_iters = 4000
    adam_lr = 1e-3
    lbfgs_iters = 400

    lambda_pde = 1.0
    lambda_bc = 1.0
    lambda_ic = 0.0
    lambda_h = 1e-3
    eps_cov = 1e-6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    pinn_means, pinn_stds = [], []
    ger_means, ger_stds = [], []

    Nx, Ny = 201, 201
    _, _, U_true = analytic_on_grid(Nx=Nx, Ny=Ny, device=device)

    print(f"\ Robustness Benchmark (averaged over {num_trials} trials).")
    print(f"{'Noise':<8} | {'PINN Avg L2':<12} | {'GERPINN Avg L2':<12}")
    print("-" * 37)

    for i, nl in enumerate(noise_levels):
        
        errors_pinn_temp = []
        errors_ger_temp = []
        
        for trial in range(num_trials):
            seed = 100 + i * 100 + trial
            entropy_data = make_entropy_points(device, N_g=100, seed=seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            shared = make_shared_training_points(device=device, N_f=N_f, N_bc=N_bc, noise_level=nl)

            pinn = PINN(
                hidden=hidden, depth=depth, act=act,
                lambda_pde=lambda_pde, lambda_ic=lambda_ic, lambda_bc=lambda_bc,
                seed=seed, device=device
            )
            assign_shared_points(pinn, shared)
            pinn.train_adam(iters=adam_iters, lr=adam_lr, verbose=False)
            pinn.train_lbfgs(max_iter=lbfgs_iters, verbose=False)
            
            ger = GERPINN(
                hidden=hidden, depth=depth, act=act,
                lambda_pde=lambda_pde, lambda_ic=lambda_ic, lambda_bc=lambda_bc,
                lambda_h=lambda_h, eps_cov=eps_cov, use_extended_grad=True,
                seed=seed, device=device
            )
            assign_shared_points(ger, shared)
            ger.X_g = entropy_data

            ger.train_adam(iters=adam_iters, lr=adam_lr, verbose=False)
            ger.lambda_h = 0.0 
            ger.train_lbfgs(max_iter=lbfgs_iters, verbose=False)
            
            _, _, U_van = pinn.predict_on_grid(Nx=Nx, Ny=Ny)
            _, _, U_ger = ger.predict_on_grid(Nx=Nx, Ny=Ny)
            
            l2_van = avg_l2_error(U_van, U_true)
            l2_ger = avg_l2_error(U_ger, U_true)
            
            errors_pinn_temp.append(l2_van)
            errors_ger_temp.append(l2_ger)
            
            print(".", end="", flush=True)

        mean_pinn = np.mean(errors_pinn_temp)
        std_pinn = np.std(errors_pinn_temp)
        mean_ger = np.mean(errors_ger_temp)
        std_ger = np.std(errors_ger_temp)
        
        pinn_means.append(mean_pinn)
        pinn_stds.append(std_pinn)
        ger_means.append(mean_ger)
        ger_stds.append(std_ger)

        print(f"\r{nl*100:5.1f}%   | {mean_pinn:.4e}   | {mean_ger:.4e}")

    pinn_means = np.array(pinn_means)
    pinn_stds = np.array(pinn_stds)
    ger_means = np.array(ger_means)
    ger_stds = np.array(ger_stds)

    plt.figure(figsize=(10, 6))
    
    #PINN
    plt.plot(noise_levels, pinn_means, 'o-', label='Vanilla PINN (Avg)', color='tab:blue', linewidth=2)
    plt.fill_between(noise_levels, 
                     pinn_means - pinn_stds, 
                     pinn_means + pinn_stds, 
                     color='tab:blue', alpha=0.2, label='PINN Std Dev')

    #GERPINN
    plt.plot(noise_levels, ger_means, 's-', label='GERPINN (Avg)', color='tab:orange', linewidth=2)
    plt.fill_between(noise_levels, 
                     ger_means - ger_stds, 
                     ger_means + ger_stds, 
                     color='tab:orange', alpha=0.2, label='GERPINN Std Dev')
    
    plt.xlabel('Noise Level ($\sigma$)')
    plt.ylabel('L2 Relative Error')
    plt.title(f'PINN vs GER on varied noise')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
