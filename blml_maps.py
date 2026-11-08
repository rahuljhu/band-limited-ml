"""Input maps: a kernel unit whose coordinates are u = map(x) of some input columns (the lattice holds u). A map
declares its parameters with a type and provides its value and its vector-Jacobian product; the estimators never
know what the parameters mean:

  params             {name: (type, trainable by default)}; types: 'stiefel' (orthonormal columns: tangent Adam,
                     QR retraction), 'free' (plain Adam), 'scale' (Adam on log theta)
  k                  outputs (lattice columns)
  forward(X)         u (n, k) from the map's input columns X (n, D)
  vjp(X, V)          {name: sum_i V_i . d u_i / d theta} for every parameter (V (n, k): the score's derivative in
                     the points' coordinates); the streaming fit moves the trainable ones by adam_step
  get(name), set(name, value)

The score's derivative in the coordinates comes from the 1-D kernels on the lattice columns (blml_local.FusedFit
.batch_stats, the 'shift' derivative), so any map whose outputs carry sinc columns trains the same way:
LinearMap (u = x W; W= / k= in the modes), or a nonlinear one (e.g. a network: forward and vjp by its own autodiff)."""
import numpy as np


def adam_step(value, grad, state, ptype, lr, betas, dlog=False):
    """One Adam ascent step of a parameter of type ptype from the gradient grad (same shape); state = [m, v, t],
    updated in place. 'stiefel': on the tangent space with one second moment for the whole step (rotation-invariant),
    then a QR retraction; 'free': elementwise Adam; 'scale': elementwise Adam on log(value) (dlog: grad is already
    d / dlog theta, as the kernels' parameter gradients)."""
    b1, b2 = betas
    value = np.asarray(value, float); grad = np.asarray(grad, float)
    if ptype == 'stiefel':
        Gt = grad - value @ ((value.T @ grad + grad.T @ value) / 2)
        state[2] += 1; t = state[2]
        state[0] = b1 * state[0] + (1 - b1) * Gt; state[1] = b2 * state[1] + (1 - b2) * float(np.sum(Gt * Gt))
        return np.linalg.qr(value + lr * (state[0] / (1 - b1 ** t)) / (np.sqrt(state[1] / (1 - b2 ** t)) + 1e-12))[0]
    if ptype == 'scale' and not dlog:
        grad = grad * value                                # d / dlog theta
    state[2] += 1; t = state[2]
    state[0] = b1 * state[0] + (1 - b1) * grad; state[1] = b2 * state[1] + (1 - b2) * grad * grad
    step = lr * (state[0] / (1 - b1 ** t)) / (np.sqrt(state[1] / (1 - b2 ** t)) + 1e-12)
    return value * np.exp(step) if ptype == 'scale' else value + step


class InputMap:
    params = {}

    def get(self, name):
        return getattr(self, name)

    def set(self, name, value):
        setattr(self, name, np.asarray(value, float))

    def forward(self, X):
        raise NotImplementedError

    def vjp(self, X, V):
        raise NotImplementedError


class LinearMap(InputMap):
    """u = x W, W (D, k) with orthonormal columns (directions only; the kernel's fc carry the resolution)."""
    params = {'W': ('stiefel', True)}

    def __init__(self, W):
        self.W = np.linalg.qr(np.asarray(W, float))[0]

    @property
    def k(self):
        return self.W.shape[1]

    def forward(self, X):
        return np.asarray(X, float) @ self.W

    def vjp(self, X, V):
        return {'W': np.asarray(X, float).T @ np.asarray(V, float)}

    def set(self, name, value):
        self.W = np.linalg.qr(np.asarray(value, float))[0]


class TanhMap(InputMap):
    """u = tanh(x W) (W free): a nonlinear map, to show (and test) that the estimators need only forward and vjp."""
    params = {'W': ('free', True)}

    def __init__(self, W):
        self.W = np.asarray(W, float)

    @property
    def k(self):
        return self.W.shape[1]

    def forward(self, X):
        return np.tanh(np.asarray(X, float) @ self.W)

    def vjp(self, X, V):
        X = np.asarray(X, float); t = np.tanh(X @ self.W)
        return {'W': X.T @ (np.asarray(V, float) * (1 - t * t))}
