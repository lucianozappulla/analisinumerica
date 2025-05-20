import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import necessario per plot 3D

# Soluzione analitica
def analytical_solution(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

# Genera punti casuali nel dominio [0,1]x[0,1]
def generate_collocation_points(N_f):
    x_f = torch.rand(N_f, 1).requires_grad_(True)
    t_f = torch.rand(N_f, 1).requires_grad_(True)
    return torch.cat([x_f, t_f], dim=1)

def generate_initial_conditions(N_ic):
    x_ic = torch.rand(N_ic, 1).requires_grad_(True)
    t_ic = torch.zeros_like(x_ic).requires_grad_(True)
    return torch.cat([x_ic, t_ic], dim=1)

def generate_boundary_conditions(N_bc):
    x_left = torch.zeros(N_bc // 2, 1).requires_grad_(True)
    t_left = torch.rand(N_bc // 2, 1).requires_grad_(True)

    x_right = torch.ones(N_bc // 2, 1).requires_grad_(True)
    t_right = torch.rand(N_bc // 2, 1).requires_grad_(True)

    return torch.cat([torch.cat([x_left, t_left], dim=1),
                      torch.cat([x_right, t_right], dim=1)], dim=0)

# Architettura della rete neurale (più profonda)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# Funzione di perdita con pesi bilanciati
def compute_loss(model, x_f, x_ic, x_bc):
    u_f = model(x_f)
    u_t = torch.autograd.grad(u_f.sum(), x_f, create_graph=True)[0][:, 1].unsqueeze(1)
    u_xx = torch.autograd.grad(u_t.sum(), x_f, create_graph=True)[0][:, 0].unsqueeze(1)
    f = u_t - u_xx
    loss_pde = torch.mean(f ** 2)

    u_ic_pred = model(x_ic)
    u_ic_true = torch.sin(np.pi * x_ic[:, 0]).unsqueeze(1)
    loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

    u_bc_pred = model(x_bc)
    loss_bc = torch.mean(u_bc_pred ** 2)

    loss = 500 * loss_pde + 10 * loss_ic + 10 * loss_bc
    return loss, loss_pde.item(), loss_ic.item(), loss_bc.item()

# Addestramento con solo Adam
def train(model, N_f, N_ic, N_bc, epochs=20000):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        x_f = generate_collocation_points(N_f)
        x_ic = generate_initial_conditions(N_ic)
        x_bc = generate_boundary_conditions(N_bc)

        loss, loss_pde, loss_ic, loss_bc = compute_loss(model, x_f, x_ic, x_bc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f} | "
                  f"PDE: {loss_pde:.6f}, IC: {loss_ic:.6f}, BC: {loss_bc:.6f}")

# Plot 3D della soluzione
def plot_3d_solution(model, save_path=None):
    model.eval()
    x = torch.linspace(0, 1, 100).unsqueeze(1)
    t = torch.linspace(0, 1, 100).unsqueeze(1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze())
    inputs = torch.cat([X.reshape(-1, 1), T.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        u_pred = model(inputs).reshape(100, 100).numpy()

    u_exact = analytical_solution(X.numpy(), T.numpy())

    fig = plt.figure(figsize=(14, 7))

    # 3D Plot PINN
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X.numpy(), T.numpy(), u_pred, cmap='viridis', linewidth=0, antialiased=False)
    ax1.set_title("Soluzione Approssimata (PINN)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_zlabel("u(x,t)")

    # 3D Plot Analitico
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X.numpy(), T.numpy(), u_exact, cmap='viridis', linewidth=0, antialiased=False)
    ax2.set_title("Soluzione Analitica")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_zlabel("u(x,t)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafico 3D salvato in '{save_path}'")
    else:
        plt.show()
    plt.close()

# Main
if __name__ == "__main__":
    # Parametri aumentati
    N_f = 50000   # Collocation points
    N_ic = 2000   # Condizioni iniziali
    N_bc = 2000   # Condizioni al contorno

    # Modello e ottimizzatore
    model = PINN()
    print("Inizio addestramento...")
    train(model, N_f, N_ic, N_bc, epochs=20000)
    print("Addestramento completato.")

    # Visualizzazione risultati
    plot_3d_solution(model, "grafico_3d.png")