# The least-control principle for learning at equilibrium

This repository contains the code accompaning the paper, [The least-control principle for learning at equilibrium](https://arxiv.org/abs/2207.01332). 

It contains the code for running the experiments for MNIST and CIFAR10, in resp. Table 1 and 2.

### Dependencies
To install depndencies, create a new conda environment with the following command
```
conda env create -f env.yaml
```

### Commands
Below are the commands to recreate the results from the paper.

#### MNIST Feedforward (FF) results

Backpropagation Baseline (BP):
```
python3 run_classification.py --data=mnist --meta_lr=0.001 --meta_optimizer=adam --meta_scheduler=cos --metagrad_clip=-1 \
--model=ff --num_epochs=30 --rnn_hidden_units=256 --trainer=bp
```

LCP Energy Based (LCP-EBD):
```
python3 run_classification.py --data=mnist --meta_lr=0.001 --meta_optimizer=adam --meta_scheduler=cos --metagrad_clip=-1 \
--model=ff --num_epochs=30 --rnn_hidden_units=256 --alpha=0.1 \
--solver_max_iter=800 --trainer=lcp-ebd --ebd_lr=0.01 --ebd_optim=adam --ff_init=False 
```

LCP Dynamic Inversion (LCP-DI):
```
python3 run_classification.py --data=mnist --meta_lr=0.001 --meta_optimizer=adam --meta_scheduler=cos --metagrad_clip=-1 \
--model=ff --num_epochs=30 --rnn_hidden_units=256 --alpha=0.1 --dt=1 --k_i=1 --k_p=0 \
--solver_error_threshold=1e-06 --solver_max_iter=800 --trainer=lcp-di
```

LCP Dynamic Inversion with Kollen Pollack (LCP-DI (KP)):
```
python3 run_classification.py --data=mnist --meta_lr=0.001 --meta_optimizer=adam --meta_scheduler=cos --metagrad_clip=-1 \
--model=ff --num_epochs=30 --rnn_hidden_units=256 --alpha=0.1 --dt=1 --k_i=1 --k_p=0 \
--solver_error_threshold=1e-06 --solver_max_iter=800 --trainer=lcp-kp  --kp_wd=1e-06
```

LCP Linear Feedback (LCP-LF):
```
python3 run_classification.py --data=mnist --meta_lr=0.001 --meta_optimizer=adam --meta_scheduler=cos --metagrad_clip=-1 \
--model=ff --num_epochs=30 --rnn_hidden_units=256 --alpha=0.1 --dt=1 --k_i=1 --k_p=0 \
--solver_error_threshold=1e-06 --solver_max_iter=800 --trainer=lcp-lf --wd_Q=0.01 --sigma=0.01 --tau_Q=100000 --timescale_factor=5
```


#### MNIST Recurrent neural netword (RNN) results

Recurrent Backpropagation Baseline (RBP):
```
python3 run_classification.py --data=mnist --meta_lr=0.001 --meta_optimizer=adam --meta_scheduler=cos --metagrad_clip=10 \
--model=rnn --num_epochs=30 --rnn_hidden_units=256 --solver_error_threshold=1e-04 --solver_max_iter=200 --trainer=rbp
```

LCP Energy Based (LCP-EBD):
```
python3 run_classification.py --data=mnist --meta_lr=0.001 --meta_optimizer=adam --meta_scheduler=cos --metagrad_clip=10 \
--model=rnn --num_epochs=30 --rnn_hidden_units=256 --alpha=0.1 \
--solver_max_iter=200 --trainer=lcp-ebd --ebd_lr=0.001 --ebd_optim=adam --ff_init=True 
```

LCP Dynamic Inversion (LCP-DI):
```
python3 run_classification.py --data=mnist --meta_lr=0.001 --meta_optimizer=adam --meta_scheduler=cos --metagrad_clip=10 \
--model=rnn --num_epochs=30 --rnn_hidden_units=256 --alpha=0.1 --dt=1 --k_i=1 --k_p=0 \
--solver_error_threshold=1e-06 --solver_max_iter=800 --trainer=lcp-di
```

LCP Dynamic Inversion with Kollen Pollack (LCP-DI (KP)):
```
python3 run_classification.py --data=mnist --meta_lr=0.001 --meta_optimizer=adam --meta_scheduler=cos --metagrad_clip=10 \
--model=rnn --num_epochs=30 --rnn_hidden_units=256 --alpha=0.1 --dt=1 --k_i=1 --k_p=0 \
--solver_error_threshold=1e-06 --solver_max_iter=800 --trainer=lcp-kp --kp_wd=1e-06
```

LCP Linear Feedback (LCP-LF):
```
python3 run_classification.py --data=mnist --meta_lr=0.001 --meta_optimizer=adam --meta_scheduler=cos --metagrad_clip=10 \
--model=rnn --num_epochs=30 --rnn_hidden_units=256 --alpha=0.1 --dt=1 --k_i=1 --k_p=0 \
--solver_error_threshold=1e-06 --solver_max_iter=800 --trainer=lcp-lf --wd_Q=0.01 --sigma=0.01  --tau_Q=100000 --timescale_factor=5
```

#### CIFAR-10 Feedforward (FF) results
Backpropagation Baseline (BP):
```
python3 run_classification.py --data=cifar10 --meta_lr=0.1 --meta_optimizer=sgd --meta_scheduler=cos --meta_wd=0.001 \
--metagrad_clip=-1 --model=conv-ff --num_epochs=30 --rnn_hidden_units=256 \
--solver_error_threshold=1e-4 --solver_max_iter=800 --trainer=bp
```

LCP Dynamic Inversion (LCP-DI):
```
python3 run_classification.py --data=cifar10 --meta_lr=0.1 --meta_optimizer=sgd --meta_scheduler=cos --meta_wd=0.001 \
--metagrad_clip=-1 --model=conv-ff --num_epochs=30 --rnn_hidden_units=256 --alpha=0.1 --dt=1 --k_i=0.1 --k_p=0 \
--solver_error_threshold=1e-06 --solver_max_iter=800 --trainer=lcp-di
```

LCP Dynamic Inversion with Kollen Pollack (LCP-DI (KP)):
```
python3 run_classification.py --data=cifar10 --meta_lr=0.1 --meta_optimizer=sgd --meta_scheduler=cos --meta_wd=0.001 \
--metagrad_clip=-1 --model=conv-ff --num_epochs=30 --rnn_hidden_units=256 --alpha=0.1 --dt=1 --k_i=0.1 --k_p=0 \
--solver_error_threshold=1e-06 --solver_max_iter=800 --trainer=lcp-kp --kp_wd=1e-05
```

#### CIFAR-10 Recurrent neural network (RNN) results

Recurrent Backpropagation Baseline (RBP):
```
python3 run_classification.py --data=cifar10 --meta_lr=0.03 --meta_optimizer=sgd --meta_scheduler=cos --metagrad_clip=10 \
--model=conv-rnn --num_epochs=75 --solver_max_iter=200 --trainer=rbp
```

LCP Dynamic Inversion (LCP-DI):
```
python3 run_classification.py --data=cifar10 --meta_lr=0.05 --meta_optimizer=sgd --meta_scheduler=cos --metagrad_clip=10 \
--model=conv-rnn --num_epochs=75 --solver_max_iter=800 --alpha=1 --k_i=0.3 --k_p=0 --trainer=lcp-di 
```
