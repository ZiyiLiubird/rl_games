from random import uniform
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np
import enum

from . import amp_network_builder

ENC_LOGIT_INIT_SCALE = 0.1


class LatentType(enum.Enum):
    uniform = 0
    sphere = 1


class ASEBuilder(amp_network_builder.AMPBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(amp_network_builder.AMPBuilder.Network):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.get('actions_num')
            input_shape = kwargs.get('input_shape')
            self.value_size = kwargs.get('value_size', 1)
            self.num_seqs = kwargs.get('num_seqs', 1)
            amp_input_shape = kwargs.get('amp_input_shape')
            self._ase_latent_shape = kwargs.get('ase_latent_shape')

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)

            self.load(params)

            actor_out_size, critic_out_size = self._build_actor_critic_net(input_shape, self._ase_latent_shape)

            self.value = torch.nn.Linear(critic_out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(actor_out_size, actions_num)
            '''
                for multidiscrete actions num is a tuple
            '''
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(actor_out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(actor_out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 

                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if (not self.space_config['learn_sigma']):
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                elif self.space_config['fixed_sigma']:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(actor_out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            self.actor_mlp.init_params()
            self.critic_mlp.init_params()

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.space_config['fixed_sigma']:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

            self._build_disc(amp_input_shape)
            self._build_enc(amp_input_shape)

            return

        def load(self, params):
            super().load(params)

            self._enc_units = params['enc']['units']
            self._enc_activation = params['enc']['activation']
            self._enc_initializer = params['enc']['initializer']
            self._enc_separate = params['enc']['separate']

            return

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            ase_latents = obs_dict['ase_latents']
            states = obs_dict.get('rnn_states', None)
            use_hidden_latents = obs_dict.get('use_hidden_latents', False)

            actor_outputs = self.eval_actor(obs, ase_latents, use_hidden_latents)
            value = self.eval_critic(obs, ase_latents, use_hidden_latents)

            output = actor_outputs + (value, states)

            return output
        
        def eval_actor(self, obs, ase_latents, use_hidden_latents=False):
            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            a_out = self.actor_mlp(a_out, ase_latents, use_hidden_latents)

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return

        def eval_critic(self, obs, ase_latents, use_hidden_latents=False):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)

            c_out = self.critic_mlp(c_out, ase_latents, use_hidden_latents)
            value = self.value_act(self.value(c_out))
            return value

        def get_enc_weights(self):
            weights = []

            for m in self._enc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))
            
            weights.append(torch.flatten(self._enc.weight))
            return weights
        
        def _build_actor_critic_net(self, input_shape, ase_latent_shape):
            style_units = [512, 256]
            style_dim = ase_latent_shape[-1]

            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()

            act_fn = self.activations_factory.create(self.activation)
            initializer = self.init_factory.create(**self.initializer)


        # def eval_enc(self, amp_obs):
        #     enc_mlp_out = self._env

class AMPStyleCatNet1(torch.nn.Module):
    def __init__(self, obs_size, ase_latent_size, units, activation,
                 style_units, style_dim, initializer):
        super().__init__()

        print('build amp style cat net: ', obs_size, ase_latent_size)

        self._activation = activation
        self._initializer = initializer
        self._dense_layers = []
        self._units = units
        self._style_dim = style_dim
        self._style_activation = torch.tanh

        self._style_mlp = self._build_style_mlp(style_units, ase_latent_size)
        self._style_dense = torch.nn.Linear(self._units[-1], style_dim)

        in_size = obs_size + style_dim
        
