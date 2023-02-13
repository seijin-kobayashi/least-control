import torch
import numpy as np
from abc import ABC, abstractmethod
# import wandb

def get_fp_solver(args):
    if args.solver == "vanilla":
        solver = VanillaSolver(max_iter=args.solver_max_iter, error_threshold=args.solver_error_threshold * args.dt,
                               threshold_metric=args.solver_threshold_metric)
    elif args.solver == "anderson":
        solver = AndersonSolver(m=5, lam=1e-4, beta=1.0, max_iter=args.solver_max_iter,
                                error_threshold=args.solver_error_threshold)
    else:
        return NotImplementedError("Solver {} unknown".format(args.solver))
    return solver

class FixpointSolver(ABC):

    def flatten(self, *args):
        shapes=[]
        tensors=[]
        batch_size = args[0].shape[0]
        for arg in args:
            assert arg.dtype == args[0].dtype and arg.device == args[0].device
            shapes.append(arg.shape)
            tensors.append(arg.view(batch_size, -1))
        return torch.cat(tensors, dim=1), shapes

    def unflatten(self, x0, shapes):
        result=[]
        idx=0
        for shape in shapes:
            numel=np.prod(shape[1:])
            result.append(x0[:, idx:idx+numel].view(*shape))
            idx=idx+numel
        return tuple(result)

    def get_relative_error(self, x, x_next, threshold_metric="l2"):
        assert threshold_metric in ["l2", "legacy_l2", "l_inf"]
        if threshold_metric == "l2":
            return (x - x_next).norm(p=2, dim=tuple(range(1, len(x.shape))) ).div(1e-5 + x.norm(p=2, dim=tuple(range(1, len(x.shape))) )).max().item()
        if threshold_metric == "l_inf":
            return (x - x_next).abs().div(1e-5 + x.abs()).max().item()
        if threshold_metric == "legacy_l2":
            return (x - x_next).norm(p=2).item() / (1e-5 + x.norm(p=2).item())
        return (x - x_next).abs().div(1e-5 + x.abs()).max().item()

    def get_wrapped_forward(self, dynamic_forward, shapes):
        def wrapped_forward(x):
            args = self.unflatten(x, shapes)
            next_args = dynamic_forward(*args)
            return self.flatten(*next_args)[0]
        return wrapped_forward

    def __call__(self, dynamic_forward, *args):
        x0, shapes = self.flatten(*args)
        x0, res = self._solve(self.get_wrapped_forward(dynamic_forward, shapes), x0)

        self.log(res)

        return self.unflatten(x0, shapes)

    @abstractmethod
    def _solve(self, dynamic_forward, x0):
        return NotImplementedError()

class VanillaSolver(FixpointSolver):
    def __init__(self, max_iter=50, error_threshold=1e-3, threshold_metric="l2"):
        self.max_iter=max_iter
        self.error_threshold=error_threshold
        self.threshold_metric=threshold_metric
        self.inner_loop_length = 0
        self.inner_loop_num = 0

    def log(self,res):
        self.inner_loop_length += len(res)
        self.inner_loop_num += 1

        # wandb.log({"relative_residual": res[-1], "len_inner_loop": len(res),
        #            "avr_len_inner_loop": self.inner_loop_length/self.inner_loop_num})
    def _solve(self, dynamic_forward, x0):
        res = []
        for _ in range(self.max_iter):
            x0_next = dynamic_forward(x0)
            res.append(self.get_relative_error(x0, x0_next, threshold_metric=self.threshold_metric))
            x0 = x0_next
            if res[-1] < self.error_threshold:
                break
        return x0, res

class AndersonSolver(FixpointSolver):
    def __init__(self, m=5, lam=1e-4, beta=1.0, max_iter=50, error_threshold=1e-2):
        self.m=m
        self.lam=lam
        self.beta=beta
        self.max_iter=max_iter
        self.error_threshold=error_threshold

    def log(self, res):
        pass

    def _solve(self, dynamic_forward, x0):
        """ Anderson acceleration for fixed point iteration. """
        bsz, d = x0.shape
        X = torch.zeros(bsz, self.m, d, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, self.m, d, dtype=x0.dtype, device=x0.device)
        X[:, 0], F[:, 0] = x0, dynamic_forward(x0)
        X[:, 1], F[:, 1] = F[:, 0], dynamic_forward(F[:, 0])

        H = torch.zeros(bsz, self.m + 1, self.m + 1, dtype=x0.dtype, device=x0.device)
        H[:, 0, 1:] = H[:, 1:, 0] = 1
        y = torch.zeros(bsz, self.m + 1, 1, dtype=x0.dtype, device=x0.device)
        y[:, 0] = 1

        res = []
        for k in range(2, self.max_iter):
            n = min(k, self.m)
            G = F[:, :n] - X[:, :n]
            H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + self.lam * \
                                     torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
            alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

            X[:, k % self.m] = self.beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - self.beta) * (alpha[:, None] @ X[:, :n])[:, 0]
            F[:, k % self.m] = dynamic_forward(X[:, k % self.m])
            res.append(self.get_relative_error(F[:, k % self.m], X[:, k % self.m]))
            if (res[-1] < self.error_threshold):
                break
        return X[:, k % self.m], res
