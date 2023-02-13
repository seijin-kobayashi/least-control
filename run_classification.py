import fixpoint_solver
import implicit_model
import fixpoint_trainer as fixpoint_trainer
from utils.data_utils import get_data
from utils.misc_utils import get_accuracy, log, str2bool
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import os
from os import path
import argparse
import wandb

def train(args, model, auxiliary_module, trainer, optimizer, scheduler, loss_fn, train_loader, device):
    for i, (X, y) in enumerate(train_loader):
        x, y = X.to(device), y.to(device)
        yp = trainer.forward_train(model, x, y, auxiliary_module=auxiliary_module)
        optimizer.zero_grad()
        loss_train = loss_fn(yp, y)
        loss_train.backward()
        if args.metagrad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.metagrad_clip)
            torch.nn.utils.clip_grad_norm_(auxiliary_module.parameters(), max_norm=args.metagrad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

def eval(model, trainer, loss_fn, dloader, device):
    log_dict=dict()
    total_loss, total_acc = 0., 0.
    for X, y in dloader:
        X, y = X.to(device), y.to(device)
        yp = trainer.forward(model, X)
        total_acc += get_accuracy(yp, y)
        total_loss += loss_fn(yp,y).item()

    log_dict["loss"]=total_loss/len(dloader)
    log_dict["accuracy"]=total_acc/len(dloader)
    return log_dict

def main():
    parser = argparse.ArgumentParser()

    # Setup
    parser.add_argument('--seed', type=int, default=-1, metavar='seed')
    parser.add_argument('--data', type=str, default="mnist", metavar='data')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--eval_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out_dir', type=str, default='./out',
                        help='Output directory.')
    parser.add_argument('--save_model', type=str2bool, default=False)

    # Solver
    parser.add_argument('--solver', type=str, default="vanilla", metavar='solver')
    parser.add_argument('--solver_max_iter', type=int, default=50, metavar = "iter")
    parser.add_argument('--solver_error_threshold', type=float, default=1e-4, metavar = "thr")
    parser.add_argument('--solver_threshold_metric', type=str, default="l2", metavar = "metric")

    # Trainer & Dynamic
    parser.add_argument('--trainer', type=str, default="lcp-di", metavar='trainer')
    parser.add_argument('--k_p', type=float, default=0., metavar='kp')
    parser.add_argument('--k_i', type=float, default=1, metavar='ki')
    parser.add_argument('--alpha', type=float, default=0., metavar='alpha')
    parser.add_argument('--dt', type=float, default=1, metavar='dt')
    parser.add_argument('--tau_v', type=float, default=1, metavar='tv')
    parser.add_argument('--tau_u', type=float, default=1, metavar='tu')
    parser.add_argument('--tau_e', type=float, default=1, metavar='te')

    # Model
    parser.add_argument('--model', type=str, default="vanilla", help="rnn, kp_rnn, conv", metavar='model')
    parser.add_argument('--rnn_hidden_units', type=int, default=256, metavar='hidden')

    # Linear Feedback Model
    parser.add_argument('--timescale_factor', type=float, default=5)
    parser.add_argument('--tau_eps', type=float, default=1)
    parser.add_argument('--tau_Q', type=float, default=100)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--wd_Q', type=float, default=0.001)

    # Kollen Pollack Model
    parser.add_argument('--kp_wd', type=float, default=0.001)

    # EBD Model
    parser.add_argument('--ebd_optim', type=str, default="adam")
    parser.add_argument('--ebd_lr', type=float, default=0.01)
    parser.add_argument('--ff_init', type=str2bool, default=False)

    # Optimization
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='Learning rate.', metavar='metalr')
    parser.add_argument('--meta_wd', type=float, default=0.,
                        help='Weight decay.', metavar='metawd')
    parser.add_argument('--meta_optimizer', type=str, default="adam", metavar='metaoptim')
    parser.add_argument('--meta_scheduler', type=str, default="none", metavar='metasched')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='gam')
    parser.add_argument('--metagrad_clip', type=float, default=-1, metavar='clip')

    args = parser.parse_args()
    wandb.init(config=args, mode="offline")
    args=wandb.config

    if not path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # SEED
    if args.seed >=0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()

    train_loader, test_loader, input_shape, output_dim = get_data(args)
    model = implicit_model.get_impliciti_model(args, input_shape, output_dim).to(device)
    solver = fixpoint_solver.get_fp_solver(args)
    trainer = fixpoint_trainer.get_fp_trainer(loss_fn, solver, args)
    auxiliary_module = trainer.get_auxiliary_module(model, device)

    if args.meta_optimizer == "adam":
        optimizer = optim.Adam(list(model.parameters())+list(auxiliary_module.parameters()), args.meta_lr, weight_decay=args.meta_wd)
    elif args.meta_optimizer == 'sgd':
        optimizer = optim.SGD(list(model.parameters())+list(auxiliary_module.parameters()), args.meta_lr, weight_decay=args.meta_wd)

    if args.meta_scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs*len(train_loader), eta_min=args.meta_lr*args.gamma)
    else:
        scheduler=None

    for epoch in range(args.num_epochs):
        train(args, model, auxiliary_module, trainer, optimizer, scheduler, loss_fn, train_loader, device)

        if args.eval_epochs > 0 and (epoch+1) % args.eval_epochs == 0:
            print("Evaluating after training epoch {}".format(epoch))
            log_dict = eval(model, trainer, loss_fn, test_loader, device)
            log_dict["epoch"]=epoch
            log(log_dict, prefix="eval_test/")

    print("Evaluating after training epoch {}".format(epoch))
    log_dict = eval(model, trainer, loss_fn, train_loader, device)
    log_dict["epoch"]=epoch
    log(log_dict, prefix="eval_train/")
    log_dict = eval(model, trainer, loss_fn, test_loader, device)
    log_dict["epoch"]=epoch
    log(log_dict, prefix="eval_test/")

    if args.save_model:
        save_path = args.out_dir + "/model.pth"
        torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    main()

