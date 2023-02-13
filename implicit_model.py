import torch
import numpy as np
from torch import nn

def get_impliciti_model(args, input_shape, output_dim):
    if args.model == "rnn":
        model = RNNModel(input_shape, args.rnn_hidden_units, output_dim)
    elif args.model == 'ff':
        model = FeedforwardModel(input_shape, args.rnn_hidden_units, output_dim)
    elif args.model == 'conv-rnn':
        model = RNNConvModel(input_shape, output_dim)
    elif args.model == 'conv-ff':
        model = FeedforwardConvModel(input_shape, output_dim)
    else:
        raise NotImplementedError("Model {} unknown".format(args.model))
    return model

class EquilibriumModule(nn.Module):
    """
    Module containing an encoder, an implicit module and a decoder.
    The forward pass is implemented by the trainer, but should look like
    output = decoder(implicit_module_*(encoder(x)))
    where implicit_module_* is partial(implicit_module, x=encoder(x)) is recursively applied to v until convergence.
    """
    @property
    def encoder_module(self):
        """ Maps the input to a hidden activation of dimension hidden_dim """
        return self._encoder_module
    @property
    def implicit_module(self):
        """ Recursively transforms the hidden activation until convergence """
        return self._implicit_module
    @property
    def decoder_module(self):
        """ Decodes some of the hidden activations (of dimension hidden_readout_dim) to the output, on which the loss can be computed """
        return self._decoder_module
    @property
    def hidden_dim(self):
        return self._hidden_dim
    @property
    def hidden_readout_dim(self):
        return self._hidden_readout_dim
    @property
    def output_dim(self):
        return self._output_dim

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view([x.shape[0], *self.shape])
class ZeroPadding(nn.Module):
    def __init__(self, padding):
        super(ZeroPadding, self).__init__()
        self.padding = padding

    def forward(self, x):
        return torch.nn.functional.pad(x, self.padding, value=0)

class Slice(nn.Module):
    def __init__(self, from_idx, to_idx):
        super(Slice, self).__init__()
        self.from_idx = from_idx
        self.to_idx = to_idx

    def forward(self, x):
        return x[:, self.from_idx:self.to_idx]

class LinearReadoutImplicitModule(nn.Module):
    def __init__(self, input_idx, output_idx, total_dim, act, d_act, bias=True, normalize=False):
        super(LinearReadoutImplicitModule, self).__init__()
        modules=[]
        self.input_idx=input_idx
        self.output_idx=output_idx
        self.total_dim=total_dim
        self._act=act
        self._d_act=d_act
        self.bias=bias
        assert len(input_idx)==len(output_idx)
        for i in range(len(input_idx)):
            in_dim=input_idx[i][1]-input_idx[i][0]
            out_dim=output_idx[i][1]-output_idx[i][0]
            modules.append(nn.Sequential(Slice(input_idx[i][0], input_idx[i][1]),
                                         nn.Linear(in_dim, out_dim, bias=bias),
                                         ZeroPadding((output_idx[i][0] , total_dim-output_idx[i][1]))))
            if normalize:
                modules[-1][1].weight.data /=  modules[-1][1].weight.norm()
        self.module_list=nn.ModuleList(modules)

    @property
    def act(self):
        return self._act
    @property
    def d_act(self):
        return self._d_act

    def get_feedback(self):
        res = LinearReadoutImplicitModule(self.output_idx, self.input_idx, self.total_dim, lambda x:x, None, bias=False, normalize=True)
        return res

    def forward(self, x, v):
        hidden = self.act(v)
        result=x
        for m in self.module_list:
            result = result + m(hidden)
        return result

class RNNModel(EquilibriumModule):
    def __init__(self, input_shape, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.input_dim = np.prod(input_shape)
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._hidden_readout_dim = hidden_dim

        act = nn.Tanh()
        d_act=lambda v: 1 - torch.tanh(v).pow(2)
        self._encoder_module=nn.Sequential(nn.Flatten(), nn.Linear(self.input_dim, hidden_dim))
        self._implicit_module=LinearReadoutImplicitModule(input_idx=[(0, hidden_dim)],
                                              output_idx=[(0, hidden_dim)],
                                              total_dim=hidden_dim, act=act, d_act=d_act)
        self._decoder_module=nn.Sequential(act, Slice(0, hidden_dim), nn.Linear(self.hidden_readout_dim, self.output_dim))

class FeedforwardModel(EquilibriumModule):
    def __init__(self, input_shape, hidden_dim, output_dim):
        super(FeedforwardModel, self).__init__()
        self.input_dim = np.prod(input_shape)
        self._hidden_dim = hidden_dim+hidden_dim
        self._output_dim = output_dim
        self._hidden_readout_dim = hidden_dim

        act = nn.Tanh()
        d_act=lambda v: 1 - torch.tanh(v).pow(2)
        self._encoder_module=nn.Sequential(nn.Flatten(), nn.Linear(self.input_dim, hidden_dim), ZeroPadding((0 , hidden_dim)))
        self._implicit_module=LinearReadoutImplicitModule(input_idx=[(0, hidden_dim)],
                                              output_idx=[(hidden_dim, 2*hidden_dim)],
                                              total_dim=self.hidden_dim, act=act, d_act=d_act)
        self._decoder_module=nn.Sequential(act, Slice(hidden_dim, 2*hidden_dim), nn.Linear(self.hidden_readout_dim, self.output_dim))

class FatConvReadoutImplicitModule(nn.Module):
    def __init__(self, feedback=False):
        super(FatConvReadoutImplicitModule, self).__init__()
        modules=[]
        self.total_dim=15*15*96+7*7*128 + 3*3*256+2*2048
        self._d_act=lambda v: v > 0

        if not feedback:
            self._act=nn.ReLU()
            in_begin=0
            in_end=15*15*96
            out_begin=in_end
            out_end=in_end+7*7*128
            modules.append(nn.Sequential(nn.Sequential(Slice(in_begin, in_end), View([96,15,15])),
                                         nn.Conv2d(96, 128, kernel_size=5, stride=2, padding=1),
                                         View((-1,)),
                                         ZeroPadding((out_begin, self.total_dim-out_end))))

            in_begin=out_begin
            in_end=out_end
            out_begin=in_end
            out_end=out_end+3*3*256
            modules.append(nn.Sequential(nn.Sequential(Slice(in_begin, in_end), View([128,7,7])),
                                         nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),
                                         View((-1,)),
                                         ZeroPadding((out_begin, self.total_dim-out_end))))

            in_begin=out_begin
            in_end=out_end
            out_begin=in_end
            out_end=out_end+2048
            modules.append(nn.Sequential(Slice(in_begin, in_end),
                                         nn.Linear(3*3*256, 2048),
                                         ZeroPadding((out_begin, self.total_dim-out_end))))

            in_begin=out_begin
            in_end=out_end
            out_begin=in_end
            out_end=out_end+2048
            modules.append(nn.Sequential(Slice(in_begin, in_end),
                                         nn.Linear(2048, 2048),
                                         ZeroPadding((out_begin, self.total_dim-out_end))))
        else:
            self._act = (lambda x:x)
            in_begin=0
            in_end=15*15*96
            out_begin=in_end
            out_end=in_end+7*7*128
            modules.append(nn.Sequential(nn.Sequential(Slice(out_begin, out_end), View([128,7,7])),
                                         nn.ConvTranspose2d(128,96, kernel_size=5, stride=2, padding=1, bias=False),
                                         View((-1,)),
                                         ZeroPadding((in_begin, self.total_dim-in_end))))

            modules[-1][1].weight.data /=  modules[-1][1].weight.norm()


            in_begin=out_begin
            in_end=out_end
            out_begin=in_end
            out_end=out_end+3*3*256
            modules.append(nn.Sequential(nn.Sequential(Slice(out_begin, out_end), View([256,3,3])),
                                         nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=1, bias=False),
                                         View((-1,)),
                                         ZeroPadding((in_begin, self.total_dim-in_end))))
            modules[-1][1].weight.data /=  modules[-1][1].weight.norm()

            in_begin=out_begin
            in_end=out_end
            out_begin=in_end
            out_end=out_end+2048
            modules.append(nn.Sequential(Slice(out_begin, out_end),
                                         nn.Linear(2048, 3*3*256, bias=False),
                                         ZeroPadding((in_begin, self.total_dim-in_end))))
            modules[-1][1].weight.data /=  modules[-1][1].weight.norm()

            in_begin=out_begin
            in_end=out_end
            out_begin=in_end
            out_end=out_end+2048
            modules.append(nn.Sequential(Slice(out_begin, out_end),
                                         nn.Linear(2048, 2048, bias=False),
                                         ZeroPadding((in_begin, self.total_dim-in_end))))
            modules[-1][1].weight.data /=  modules[-1][1].weight.norm()

        self.module_list=nn.ModuleList(modules)

    @property
    def act(self):
        return self._act

    @property
    def d_act(self):
        return self._d_act

    def get_feedback(self):
        return FatConvReadoutImplicitModule(feedback=True)

    def forward(self, x, v):
        hidden = self.act(v)
        result=x
        for m in self.module_list:
            result = result + m(hidden)
        return result

class FeedforwardConvModel(EquilibriumModule):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.input_shape = input_shape
        self._hidden_readout_dim = 2048
        self._output_dim = output_dim

        self._implicit_module = FatConvReadoutImplicitModule()
        self._encoder_module = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, kernel_size=5, padding=2),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            View((-1,)),
            ZeroPadding((0, self.implicit_module.total_dim - 15*15*96)))
        self._decoder_module=nn.Sequential(nn.ReLU(), Slice(self.implicit_module.total_dim-self.hidden_readout_dim, self.implicit_module.total_dim), nn.Linear(self.hidden_readout_dim, output_dim))
        self._hidden_dim = self.implicit_module.total_dim

class ResNetImplicitModule(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self._act = nn.ReLU()
        self._d_act=lambda v: v > 0
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    @property
    def act(self):
        return self._act
    @property
    def d_act(self):
        return self._d_act

    def forward(self, z, x):
        y = self.norm1(self.act(self.conv1(z)))
        return self.norm3(self.act(z + self.norm2(x + self.conv2(y))))

class RNNConvModel(EquilibriumModule):
    """
    This class does not support working with trainer other than RBP and LCP-DI.
    """
    def __init__(self, input_shape, output_dim):
        super(RNNConvModel, self).__init__()

        chan = 48
        self._encoder_module=nn.Sequential(
                      nn.Conv2d(input_shape[0],chan, kernel_size=3, bias=True, padding=1),
                      nn.BatchNorm2d(chan))
        self._implicit_module=ResNetImplicitModule(chan, 64, kernel_size=3)
        self._decoder_module=nn.Sequential(nn.BatchNorm2d(chan),
                      nn.AvgPool2d(8,8),
                      nn.Flatten(),
                      nn.Linear(chan*(input_shape[1]//8)**2,output_dim))

    @property
    def hidden_dim(self):
        raise NotImplementedError("Only lcp-di or rbp supported by RNNConvModel")
    @property
    def hidden_readout_dim(self):
        raise NotImplementedError("Only lcp-di or rbp supported by RNNConvModel")
    @property
    def output_dim(self):
        raise NotImplementedError("Only lcp-di or rbp supported by RNNConvModel")