import math
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
import torch
import wandb

def get_fp_trainer(loss_fn, root_finder, args):
    if args.trainer == 'rbp':
        trainer = RBP(loss_fn, root_finder, args)
    elif args.trainer == 'bp':
        trainer = BP(loss_fn, root_finder, args)
    elif args.trainer == 'lcp-di':
        trainer = LCPDynamicInversion(loss_fn, root_finder, args)
    elif args.trainer == 'lcp-kp':
        trainer = LCPKollenPollack(loss_fn, root_finder, args)
    elif args.trainer == 'lcp-lf':
        trainer = LCPLinearFeedback(loss_fn, root_finder, args)
    elif args.trainer == 'lcp-ebd':
        trainer = EnergyEBDTrainer(loss_fn, root_finder, args)
    else:
        raise NotImplementedError("Trainer {} unknown".format(args.trainer))
    return trainer

class FixpointTrainer(nn.Module):
    def __init__(self, loss_fn, root_finder, args):
        super(FixpointTrainer, self).__init__()
        self.loss_fn=loss_fn
        self.root_finder=root_finder
        self.args=args

    def get_auxiliary_module(self, model, device):
        class EmptyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
        return EmptyModule()

    def next_state_fn(self, model, x):
        return lambda v: (model(x, v).detach(),)

    def find_fixpoint(self, model, encoding):
        v = torch.zeros_like(encoding)
        with torch.no_grad():
            return self.root_finder(self.next_state_fn(model.implicit_module, encoding), v)[0]

    def forward(self, model, x):
        encoding = model.encoder_module(x)
        v_star = self.find_fixpoint(model, encoding)
        decoding = model.decoder_module(v_star)
        return decoding

    @abstractmethod
    def forward_train(self, model, x, y, **kwargs):
        return NotImplementedError()

class BP(FixpointTrainer):
    def forward(self, model, x):
        encoding = model.encoder_module(x)
        v = torch.zeros_like(encoding)
        for _ in range(len(model.implicit_module.module_list)+1):
            v = model.implicit_module(encoding, v)
        decoding = model.decoder_module(v)
        return decoding

    @abstractmethod
    def forward_train(self, model, x, y, **kwargs):
        return self.forward(model, x)

class RBP(FixpointTrainer):
    def forward_train(self, model, x, y, **kwargs):
        encoding = model.encoder_module(x)
        v_star = self.find_fixpoint(model, encoding)

        # Attach to graph
        v_star = model.implicit_module(encoding, v_star)
        decoding = model.decoder_module(v_star)

        v0 = v_star.clone().detach().requires_grad_(True)
        f0 = model.implicit_module(encoding, v0)

        def backward_hook(grad):
            gd = self.root_finder(lambda g : (torch.autograd.grad(f0, v0, g, retain_graph=True)[0] + grad,), grad)[0]
            return gd
        v_star.register_hook(backward_hook)
        return decoding

class LCPFixpointTrainer(FixpointTrainer):

    def next_controlled_state_fn(self, implicit_module, decoder_module, x, y, **kwargs):
        raise NotImplementedError()

    def find_controlled_fixpoint(self, model, encoding, y, **kwargs):
        v = torch.zeros_like(encoding)
        u_i = torch.zeros_like(model.decoder_module(v))
        e = torch.zeros_like(encoding)
        v_pred = torch.zeros_like(u_i)
        return self.root_finder(self.next_controlled_state_fn(model.implicit_module, model.decoder_module, encoding, y, **kwargs),
                               v, v_pred, u_i, e)

class LCPDynamicInversion(LCPFixpointTrainer):
    def next_controlled_state_fn(self, implicit_module, decoder_module, x, y, **kwargs):
        def get_next_controlled_state(v, v_pred, u_i, e):
            v.requires_grad_()
            v_pred.requires_grad_()
            grad_pred = torch.autograd.grad(self.loss_fn(v_pred, y), v_pred)[0]
            u_p = grad_pred
            u = self.args.k_i * u_i - self.args.k_p * u_p
            next_e = e + self.args.dt/self.args.tau_e*(torch.autograd.grad(implicit_module(x, v)-v, v, e)[0] +
                                                   torch.autograd.grad(decoder_module(v), v, u)[0])
            next_v = v + self.args.dt/self.args.tau_v*(implicit_module(x, v) -v + e)
            next_v_pred = v_pred + self.args.dt/self.args.tau_v*(decoder_module(v) - v_pred + u)
            next_u_i = u_i + self.args.dt/self.args.tau_u*(- grad_pred - self.args.alpha * u_i)
            return next_v.detach(), next_v_pred.detach(), next_u_i.detach(), next_e.detach()
        return get_next_controlled_state

    def forward_train(self, model,  x, y, **kwargs):
        encoding = model.encoder_module(x)
        v_star, v_pred_star, u_i_star, e_star = self.find_controlled_fixpoint(model, encoding, y)

        # Compute u*
        v_pred_star.requires_grad_()
        u_p_star = torch.autograd.grad(self.loss_fn(v_pred_star, y), v_pred_star)[0]
        u_star = (self.args.k_i * u_i_star - self.args.k_p * u_p_star).detach()

        # Attach to graph
        v_star=model.implicit_module(encoding, v_star) + e_star
        v_pred_star = (model.decoder_module(v_star) + u_star)

        v_star.register_hook(lambda _: -e_star)
        v_pred_star.register_hook(lambda _: -u_star)
        return v_pred_star

class LCPKollenPollack(LCPDynamicInversion):
    def get_auxiliary_module(self, model, device):
        class FeedbackModule(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.implicit_module=model.implicit_module.get_feedback().to(device)
                last_hidden_dim = model.hidden_readout_dim
                self.D_t=nn.Parameter(torch.randn(last_hidden_dim, model.output_dim)/math.sqrt(model.output_dim*last_hidden_dim))
                self.decoder_module = lambda x: torch.nn.functional.pad(F.linear(x, self.D_t),(model.hidden_dim-last_hidden_dim,0), value=0)
        return FeedbackModule(model).to(device)

    def decay_weights(self, model, feedback_model):
        for p in feedback_model.parameters():
            p.data-= self.args.kp_wd*p.data
        for p in model.implicit_module.parameters():
            p.data-= self.args.kp_wd*p.data
        for p in model.decoder_module.parameters():
            p.data-= self.args.kp_wd*p.data

    def next_controlled_state_fn(self, implicit_module, decoder_module, x, y, auxiliary_module=None, d_act=None, **kwargs):
        def get_next_controlled_state(v, v_pred, u_i, e):
            v.requires_grad_()
            v_pred.requires_grad_()
            grad_pred = torch.autograd.grad(self.loss_fn(v_pred, y), v_pred)[0]
            u_p = grad_pred
            u = self.args.k_i * u_i - self.args.k_p * u_p
            d_v = d_act(v)

            next_e = e + self.args.dt/self.args.tau_e*(-e + d_v.mul(auxiliary_module.implicit_module(0, e) + auxiliary_module.decoder_module(u)))
            next_v = v + self.args.dt/self.args.tau_v*(implicit_module(x, v) -v + e)
            next_v_pred = v_pred + self.args.dt/self.args.tau_v*(decoder_module(v) - v_pred + u)
            next_u_i = u_i + self.args.dt/self.args.tau_u*(- grad_pred - self.args.alpha * u_i)
            return next_v.detach(), next_v_pred.detach(), next_u_i.detach(), next_e.detach()
        return get_next_controlled_state

    def forward_train(self, model,  x, y, auxiliary_module=None):
        # Decay weight
        self.decay_weights(model, auxiliary_module)

        encoding = model.encoder_module(x)
        v_star, v_pred_star, u_i_star, e_star = self.find_controlled_fixpoint(model, encoding, y,
                                                                              d_act=model.implicit_module.d_act,
                                                                              auxiliary_module=auxiliary_module)

        # Compute u*
        v_pred_star.requires_grad_()
        u_p_star = torch.autograd.grad(self.loss_fn(v_pred_star, y), v_pred_star)[0]
        u_star = (self.args.k_i * u_i_star - self.args.k_p * u_p_star).detach()

        # Attach to graph
        feedback = auxiliary_module.implicit_module(0, e_star) + auxiliary_module.decoder_module(u_star)
        feedback.register_hook(lambda _: -model.implicit_module.act(v_star))

        v_star=model.implicit_module(encoding, v_star) + e_star + feedback - feedback.detach()
        v_pred_star = (model.decoder_module(v_star) + u_star)

        v_star.register_hook(lambda _: -e_star)
        v_pred_star.register_hook(lambda _: -u_star)
        return v_pred_star


class LCPLinearFeedback(LCPFixpointTrainer):
    # This model assumes a fixed readout layer.
    def get_auxiliary_module(self, model, device):
        class QModule(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.register_buffer("Q",torch.randn(model.hidden_dim, model.output_dim)/math.sqrt(model.output_dim))
        return QModule(model).to(device)

    def next_controlled_state_fn(self, implicit_module, decoder_module, x, y, auxiliary_module=None, **kwargs):
        def get_next_controlled_state(v, v_pred, u_i, e):
            v.requires_grad_()
            v_pred.requires_grad_()
            grad_pred = torch.autograd.grad(self.loss_fn(v_pred, y), v_pred)[0]
            u_p = grad_pred
            u = self.args.k_i * u_i - self.args.k_p * u_p
            next_e = F.linear(u, auxiliary_module.Q)
            next_v = v + self.args.dt/self.args.tau_v*(implicit_module(x, v) -v + next_e)
            next_v_pred = v_pred + self.args.dt/self.args.tau_v*(decoder_module(v) - v_pred + u)
            next_u_i = u_i + self.args.dt/self.args.tau_u*(- grad_pred - self.args.alpha * u_i)
            return next_v.detach(), next_v_pred.detach(), next_u_i.detach(), next_e.detach()
        return get_next_controlled_state

    def next_noisy_next_controlled_state_fn(self, implicit_module, decoder_module, x, y, Q=None):
        def get_next_controlled_state(v, v_pred, u_i, epsilon, cumul_Q, u_ref):
            # For the noisy dynamic, we introduce the appropriate time constants
            dt=self.args.dt/self.args.timescale_factor**2
            v.requires_grad_()
            v_pred.requires_grad_()
            grad_pred = torch.autograd.grad(self.loss_fn(v_pred, y), v_pred)[0]
            u_p = grad_pred
            u = self.args.k_i * u_i - self.args.k_p * u_p
            next_epsilon = epsilon+self.args.timescale_factor/self.args.tau_eps*(- dt*epsilon + math.sqrt(dt)*torch.randn_like(epsilon))
            v_fb = F.linear(u, Q)+ self.args.sigma*epsilon
            next_v = v + self.args.timescale_factor**2*dt/self.args.tau_v*(implicit_module(x, v) -v + v_fb)
            next_v_pred = v_pred + dt/self.args.tau_v*(decoder_module(v) - v_pred + u)
            next_u_i = u_i + dt/self.args.tau_u*(- grad_pred - self.args.alpha * u_i)

            next_cumul_Q = cumul_Q + 1/self.args.sigma**2*(-v_fb.unsqueeze(-1) @ (u-u_ref).unsqueeze(-2))
            return next_v.detach(), next_v_pred.detach(), next_u_i.detach(), next_epsilon.detach(), next_cumul_Q.detach(), u_ref.detach()
        return get_next_controlled_state

    def find_noisy_controlled_fixpoint(self, model, encoding, y, v, v_pred, u_i, u, Q=None, **kwargs):
        epsilon=torch.randn_like(v)
        cumul_Q=torch.stack([torch.zeros_like(Q) for _ in range(epsilon.shape[0])])
        return self.root_finder(self.next_noisy_next_controlled_state_fn(model.implicit_module, model.decoder_module, encoding, y, Q=Q, **kwargs),
                               v, v_pred, u_i, epsilon, cumul_Q, u)

    def learn_Q(self, Q, model, encoding, y, v, v_pred, u_i, u):
        # Update Q with noisy control
        _, _, _, _,Q_next,_= self.find_noisy_controlled_fixpoint(model, encoding, y, v, v_pred, u_i, u, Q=Q)
        Q_next=Q_next.mean(0).detach()
        Q.data += self.args.dt / self.args.tau_Q * (Q_next - self.args.wd_Q*Q)

    def forward_train(self, model, x, y, auxiliary_module=None):
        encoding = model.encoder_module(x)
        v_star, v_pred_star, u_i_star, e_star = self.find_controlled_fixpoint(model, encoding, y, auxiliary_module=auxiliary_module)

        # Compute u*
        v_pred_star.requires_grad_()
        u_p_star = torch.autograd.grad(self.loss_fn(v_pred_star, y), v_pred_star)[0]
        u_star = (self.args.k_i * u_i_star - self.args.k_p * u_p_star).detach()

        # Attach to graph
        v_star=model.implicit_module(encoding, v_star) + e_star
        v_pred_star = (model.decoder_module(v_star) + u_star)

        modulation = model.implicit_module.d_act(v_star)
        v_star.register_hook(lambda _: -e_star.mul(modulation))
        v_pred_star.register_hook(lambda _: -u_star)

        self.learn_Q(auxiliary_module.Q, model, encoding, y, v_star, v_pred_star, u_i_star, u_star)
        return v_pred_star

class EnergyEBDTrainer(FixpointTrainer):
    def find_controlled_fixpoint(self, model, encoding, y, **kwargs):
        encoding=encoding.detach()
        v = torch.zeros_like(encoding).requires_grad_()
        v_pred = torch.zeros_like(model.decoder_module(v)).requires_grad_()

        if self.args.ff_init :
            v = (self.find_fixpoint(model, encoding).detach()).requires_grad_()
            v_pred = (model.decoder_module(v).detach()).requires_grad_()

        if self.args.ebd_optim == "gd":
            optimizer = torch.optim.SGD([v, v_pred], self.args.ebd_lr)
        if self.args.ebd_optim == "adam":
            optimizer = torch.optim.Adam([v, v_pred], self.args.ebd_lr)

        for _ in range(self.args.solver_max_iter):
            energy = 0.5*(v-model.implicit_module(encoding, v)).pow(2).sum() + \
                     0.5*(v_pred-model.decoder_module(v)).pow(2).sum() + \
                     1/self.args.alpha*self.loss_fn(v_pred, y)

            wandb.log({"energy": energy.item()})
            optimizer.zero_grad()
            energy.backward()
            optimizer.step()
        return v.detach(), v_pred.detach()

    def forward_train(self, model,  x, y, **kwargs):
        encoding = model.encoder_module(x)
        v_star, v_pred_star = self.find_controlled_fixpoint(model, encoding, y)

        e_star= - (model.implicit_module(encoding, v_star) - v_star).detach()
        u_star = - (model.decoder_module(v_star) - v_pred_star).detach()

        v_star=model.implicit_module(encoding, v_star) + e_star
        v_pred_star = (model.decoder_module(v_star) + u_star)

        v_star.register_hook(lambda _: -e_star)
        v_pred_star.register_hook(lambda _: -u_star)
        return v_pred_star

