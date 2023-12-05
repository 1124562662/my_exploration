import argparse
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models.resnet import BasicBlock
import torch.nn.functional as F


class ResnetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding="same"),
            nn.MaxPool2d((3, 3), stride=2),
        )
        self.residual_blocks = nn.Sequential(
            BasicBlock(out_channels, out_channels),
            BasicBlock(out_channels, out_channels),
        )
        self.norm_out = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        x = self.inp(x)
        x = self.residual_blocks(x)
        x = self.norm_out(x)
        return x


class BYOLEncoder(nn.Module):
    def __init__(self, in_channels, out_size, emb_dim=10):
        super().__init__()
        self.in_channels = in_channels
        self.resnet_units = nn.Sequential(
            ResnetUnit(in_channels, 32),
            ResnetUnit(32, 32),
            ResnetUnit(32, 32),
            ResnetUnit(32, 32),
            # ResnetUnit(32, 32)
        )

        self.out = nn.Linear(out_size, emb_dim)

    def forward(self, x):
        if self.in_channels == 1 and len(x.size()) == 3:
            x = x.unsqueeze(1)
        x = self.resnet_units(x)
        x = x.flatten(start_dim=1)
        # print("resnet forward", x.size())
        x = self.out(x)
        return x


class ClosedLoopRNNCell(nn.Module):
    def __init__(self):
        super().__init__()
        pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_mlp(inp_size, num_hidden, num_units, out_size):
    layers = [nn.Linear(inp_size, num_units), nn.ReLU()]
    for _ in range(num_hidden - 1):
        layers.append(nn.Linear(num_units, num_units))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(num_units, out_size))

    return nn.Sequential(*layers)


class Embedding(nn.Module):
    def __init__(self, obs_dim, emb_dim, num_hidden, num_units, tau=0.01):
        super().__init__()
        # Value used to update the target network
        self.tau = tau

        self.net = create_mlp(obs_dim, num_hidden, num_units, emb_dim)
        self.net_tgt = create_mlp(obs_dim, num_hidden, num_units, emb_dim)

    def update_target(self):
        """Perform soft update of the target network."""
        for tgt_param, param in zip(self.net_tgt.parameters(), self.net.parameters()):
            tgt_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * tgt_param.data)

    def get_tgt_emb(self, obs):
        """Embed the observation using the target."""
        with torch.no_grad():
            return self.net_tgt(obs)

    def forward(self, obs):
        return self.net(obs)


class BYOL(nn.Module):
    def __init__(self, obs_dim, action_dim, num_hidden, num_units, emb_dim, hidden_size,tau=0.01, num_layers=30, action_e_dim = 80):
        super().__init__()
        self.embedding = Embedding(obs_dim, emb_dim, num_hidden, num_units,tau=tau)
        self.action_dim = action_dim
        self.action_e_dim = action_e_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.action_embedding = torch.nn.Embedding(num_embeddings=action_dim, embedding_dim=action_e_dim)
        self.closed_loop_rnn = torch.nn.GRU(input_size=emb_dim + action_e_dim, hidden_size=hidden_size,
                                            num_layers=num_layers,
                                            bias=True,
                                            batch_first=True)  # for the first state, initalise hn with all ones
        self.open_loop_rnn = torch.nn.GRU(input_size=action_e_dim, hidden_size=hidden_size,
                                          num_layers=num_layers,
                                          bias=True, batch_first=True)
        self.predictor = create_mlp(hidden_size, num_hidden, num_units, emb_dim)
        self.optimizer = Adam(self.parameters(), lr=1e-4)

        # params = [
        #     list(self.embedding.parameters()),
        #     list(self.generator.parameters()),
        #     list(self.critic.parameters()),
        #     list(self.world_model.parameters())
        # ]

    @torch.no_grad()
    def get_action_predictions_variances(self,obs, previous_c_hn=None, o_hiddens=None):
        # o_hiddens    (numlayers,previous states nums P,envNum, hidden size H)
        # previous_c_hn                     (num_layers, envNum EN, hidden_size)
        # obs                               (envNum ,E)

        o_pred, obs = self.get_action_predictions(obs, previous_c_hn, o_hiddens)# (previous states nums P+1, envNum, action_dim , embedding ),
                                                                                # (envNum ,E)
        # o_pred = o_pred.permute(1,2,0,3)  # ( envNum, action_dim ,previous states nums P+1, embedding )
        vars = torch.var(o_pred,dim=0).mean(2)  # (envNum, action_dim)

        return vars,obs# (envNum, action_dim) ,  (envNum ,E)

    @staticmethod
    def get_action_predictions_variances_multi_byols(byols,device, obs, previous_c_hn=None, o_hiddens=None):
        # byols        list of byol models
        # o_hiddens    (numlayers,previous states nums P,envNum, hidden size H)
        # previous_c_hn                     (num_layers, envNum EN, hidden_size)
        # obs                               (envNum ,E)
        with torch.no_grad():
            P =0
            if o_hiddens is not None:
                P =o_hiddens.size(1)
            res_size = (len(byols),P+1,obs.size(0),byols[0].action_dim,byols[0].emb_dim)
            res = torch.zeros(res_size).to(device) #  (byol num, previous states nums P+1, envNum, action_dim , embedding )
            obss = torch.zeros((len(byols), obs.size(0),byols[0].emb_dim)).to(device)  #  (byol num, envNum ,E)

            for idx,b in enumerate(byols):
                o_pred, obs = b.get_action_predictions(obs, previous_c_hn,
                                                      o_hiddens)  # (previous states nums P+1, envNum, action_dim , embedding ),(envNum ,E)
                res[idx] = o_pred
                obss[idx] = obs

            vars = torch.var(torch.var(res, dim=0),dim=0).mean(2)  # (envNum, action_dim)
            return vars, obss.mean(0)  # (envNum, action_dim) ,  (envNum ,E)



    @torch.no_grad()
    def get_action_predictions(self, obs, previous_c_hn=None, o_hiddens=None):
        # o_hiddens    (numlayers,previous states nums P,envNum, hidden size H)
        # previous_c_hn                     (num_layers, envNum EN, hidden_size)
        # obs                               (envNum ,E)
        previous_states_nums = 0
        if previous_c_hn is not None:
            previous_states_nums = o_hiddens.size(1)
        obs = self.embedding(obs)
        envs_num = obs.size(0)
        actions = torch.arange(start=0, end=self.action_dim)  # (action_nums,)
        action_emb = self.action_embedding(actions).unsqueeze(0).expand(envs_num, -1, -1)   # (envs_num, action_nums, action_emb)
        action_emb = action_emb.reshape((-1, action_emb.size(2)))  # (envs_num * action_nums, action_emb)

        obs_expand = obs.unsqueeze(1).expand(-1, self.action_dim, -1)  # (envs_num, action_nums, obs_emb)
        obs_expand = obs_expand.reshape((-1, obs_expand.size(2)))  # (envs_num * action_nums, obs_emb)
        emb = torch.concatenate((obs_expand, action_emb), dim=1).unsqueeze(1)  # (envNum * action_nums, 1, E+A)

        # print("P",previous_states_nums," envs_num",envs_num," action_nums ",self.action_dim," hidden_size",self.hidden_size)
        if previous_c_hn is not None:
            previous_c_hn = previous_c_hn.unsqueeze(2).expand(-1, -1, self.action_dim,-1)  # (num_layers, envNum EN, action_nums, hidden_size)
            previous_c_hn = previous_c_hn.reshape(
                (self.num_layers, -1, self.hidden_size))  # (numlayers, envNum * action_dim, hidden size H)
            c_out, c_hn = self.closed_loop_rnn(emb, previous_c_hn)
        else:
            c_hn_size = (self.num_layers, envs_num * self.action_dim, self.hidden_size)
            c_out, c_hn = self.closed_loop_rnn(emb, torch.ones(
                c_hn_size))  # (envNum * action_nums ,1 ,hidden_size ) , (num_layers, envNum * action_nums, hidden_size)

        if o_hiddens is None:
            o_hiddens = c_hn.unsqueeze(1)  # (num_layers, 1, envNum * action_nums, hidden_size)
        else:
            # print("o_hiddens", o_hiddens.size())
            o_hiddens = o_hiddens.unsqueeze(3).expand(-1, -1, -1, self.action_dim,
                                                      -1)  # (numlayers, previous states nums P+1, envNum , action_nums, hidden size H)
            o_hiddens = o_hiddens.reshape((self.num_layers, previous_states_nums, -1,
                                           self.hidden_size))  # (numlayers, previous states nums P, envNum * action_nums, hidden size H)
            o_hiddens = torch.concatenate((o_hiddens, c_hn.unsqueeze(1)),
                                          dim=1)  # (numlayers, previous states nums P+1, envNum * action_nums, hidden size H)  #TODO remove cat

        o_hiddens = o_hiddens.reshape(self.num_layers, -1,
                                      self.hidden_size)  # (numlayers, previous states nums P+1 * envNum * action_dim, hidden size H)
        action_emb = action_emb.unsqueeze(0).expand(previous_states_nums + 1, -1,
                                                    -1)  # (previous states nums P+1, env nums * action_nums, action_e_dim)
        action_emb = action_emb.reshape( (-1, self.action_e_dim))  # (previous states nums P+1 * env nums * action_nums, action_e_dim)
        action_emb = action_emb.unsqueeze(1)  # (previous states nums P+1 * env nums * action_nums,1,action_e_dim)
        o_out, o_hn = self.open_loop_rnn(action_emb,
                                         o_hiddens)  # (previous states nums P+1 * envNum * action_dim ,1 ,hidden_size ),
        # (num_layers,previous states nums P+1 * envNum * action_dim, hidden_size)
        o_pred = self.predictor(o_out.reshape((-1, o_out.size(2))))  # (previous states nums P+1 * envNum * action_dim , embedding )
        o_pred = o_pred.reshape((previous_states_nums + 1, envs_num, self.action_dim,
                                 -1))  # (previous states nums P+1, envNum, action_dim , embedding )
        return o_pred, obs  # (previous states nums P+1, envNum, action_dim , embedding ),  (envNum ,E)

    def get_intrinsic_reward(self, obs, action, obs_next, previous_c_hn=None, o_hiddens=None):
        # o_hiddens (numlayers,previous states nums P,envNum, hidden size H)
        obs_emb = self.embedding(obs)  # (envNum ,E)
        envNum = obs_emb.size(0)
        action_emb = self.action_embedding(action)  # (envNum ,action_dim)
        emb = torch.concatenate((obs_emb, action_emb), dim=1).unsqueeze(1)  # (envNum,1,E+A)

        if previous_c_hn is not None:
            c_out, c_hn = self.closed_loop_rnn(emb, previous_c_hn)
        else:
            c_hn_size = (self.num_layers, envNum, self.hidden_size)
            c_out, c_hn = self.closed_loop_rnn(emb, torch.ones(
                c_hn_size))  # (envNum ,1 ,hidden_size ) , (num_layers, envNum EN, hidden_size)

        if o_hiddens is None:
            previous_states_nums = 0
            o_hiddens = c_hn.unsqueeze(1).clone()  # (num_layers, 1, envNum EN, hidden_size)
        else:
            previous_states_nums = o_hiddens.size(1)
            o_hiddens = torch.concatenate((o_hiddens, c_hn.unsqueeze(1)),
                                          dim=1)  # (numlayers, previous states nums P+1, envNum, hidden size H)  #TODO remove cat

        action_emb = action_emb.unsqueeze(0).expand(previous_states_nums + 1, -1,
                                                    -1)  # (previous states nums P+1, env nums , action_dim)
        action_emb = action_emb.reshape(
            ((previous_states_nums + 1) * envNum, -1)).unsqueeze(1)   # (previous states nums P+1 * env nums, 1, action_dim)
        o_hiddens = o_hiddens.reshape(self.num_layers, -1,
                                        self.hidden_size)  # (numlayers, previous states nums P+1 * envNum, hidden size H)
        o_out, o_hn = self.open_loop_rnn(action_emb,
                                         o_hiddens)  # (previous states nums P+1 * env nums ,1 ,hidden_size ), (num_layers, previous states nums P+1 * env nums, hidden_size)
        o_hn = o_hn.reshape((self.num_layers,previous_states_nums+1,envNum,self.hidden_size)) #(numlayers,previous states nums P + 1 ,envNum, hidden size H)
        # Calculate the intrinsic reward loss
        o_pred = self.predictor(o_out.reshape((-1, o_out.size(2))))  # (envNum *(P+1),E)
        obs_emb_next = self.embedding.get_tgt_emb(obs_next)  # (envNum, E)
        obs_emb_next = obs_emb_next.unsqueeze(0).expand(previous_states_nums + 1, -1, -1).reshape(
            (-1, obs_emb_next.size(1)))  # (envNum *(P+1),E)
        intrinsic_loss = F.mse_loss(o_pred, obs_emb_next, reduction='none').reshape(
            (envNum, -1))  # (envNum ,(P+1)), all the partial open loop prediction
        intrinsic_loss = intrinsic_loss.mean(dim=1) / float(o_pred.size(0))  # (envNum ,)

        return o_hn, \
               intrinsic_loss, \
               c_hn

    def train_byol(self, args, device,
                   b_obs, b_actions, byol_encoder,
                   steps:int,
                   c_hn_t,  # closed_loop_hn_for_training (steps, byol num layers, env nums, byol hidden size)
                   last_ep_c_hn, # ( num_layers, num_envs ,  hidden_size),
                   o_minibatch=5,
                   ):
        horizon = args.byol_open_loop_gru_max_horizon
        envNum = last_ep_c_hn.size(1)
        c_hn = c_hn_t.to(device)

        for epoch in range(args.byol_epochs):
            obs = byol_encoder(b_obs) # (steps  * env nums ,resnet out)
            obs_embs = self.embedding(obs)  # (steps  * env nums ,E)
            actions_embs = self.action_embedding(b_actions).reshape(
                (args.num_steps, args.num_envs, -1))  # (steps, env nums, action_emb)
            actions_embs_ = actions_embs[:-1, :, :]  # (steps-1, env nums, action_emb)
            obs_embs_input = obs_embs.reshape((steps, envNum, -1))[:-1, :, :]  # (steps-1,  env nums, byol_emb)
            with torch.no_grad():
                # build target
                obs_embs_t = self.embedding.get_tgt_emb(obs)  # (steps  * env nums ,E)
                obs_embs_target = obs_embs_t.reshape((steps, envNum, -1))[1:, :, :]  # (steps-1,  env nums, byol_emb)

            # train closed loop GRU
            input_c = torch.concatenate((obs_embs_input, actions_embs_), dim=-1).reshape(
                (args.num_steps - 1, args.num_envs, -1)).transpose(0, 1)  # (env nums, steps-1, byol_emb+action_emb) #TODO remove cat
            obs_embs_pred, _ = self.closed_loop_rnn(input_c,
                                                    last_ep_c_hn)  # ( env nums,steps-1,  byol_hidden),  ( num_layers,env nums, byol_hidden)
            c_pred = self.predictor(obs_embs_pred) # ( env nums,steps-1,  byol emb)
            loss_c = F.mse_loss(c_pred.transpose(0, 1), obs_embs_target.detach(),reduction='mean')

            # train open loop GRU
            train_start = torch.randint(low=0, high= steps-horizon-1, size=(o_minibatch,args.num_envs),
                                        dtype=torch.int64)  # ( minibatch, num_envs)
            # chn is (steps, byol num layers, env nums, byol hidden size)
            # (o_minibatch, steps, byol num layers, env nums, byol hidden size)
            chn_view = c_hn.unsqueeze(0).expand(o_minibatch,-1,-1,-1,-1).permute(1,0,3,2,4) # ( steps, o_minibatch, env nums , byol num layers,  byol hidden size)
            chn_view = chn_view.reshape((-1,self.num_layers, self.hidden_size)) # ( steps * o_minibatch * env nums , num layers, hidden size)
            chn_start = train_start.reshape(-1) + torch.arange(start=0,end=steps * o_minibatch * envNum,step=steps,dtype=torch.int64)   # (  minibatch * num_envs)
            chn_view = chn_view[chn_start,:,:].reshape((o_minibatch * envNum,self.num_layers,self.hidden_size)).transpose(0,1) # ( num_layers, o_minibatch*envNum, hidden_size)

            # clone() is important
            indices = train_start.unsqueeze(2).expand(-1, -1,
                                                      args.num_steps).clone()# (  minibatch, env nums, num_steps)
            indices[:, :, horizon:] = 0  # (  minibatch, env nums, num_steps)
            ascending_li = torch.arange(0, horizon,dtype=torch.int64).to(device)  # ( horizon,)
            ascending_li = torch.cat((ascending_li, torch.zeros((args.num_steps - horizon),dtype=torch.int64)), dim=0)  # ( num_steps) #TODO remove cat
            a_indices = indices + ascending_li.unsqueeze(0).unsqueeze(0).expand(o_minibatch, args.num_envs,
                                                                              -1)   # (  minibatch, args.num_envs,  num_steps)
            a_epoch = actions_embs.unsqueeze(1).expand(-1,o_minibatch,-1,-1) # (steps, o_minibatch, env nums, action_emb)
            a_epoch = a_epoch.reshape((a_epoch.size(0), -1, self.action_e_dim)).transpose(0,1) # ( o_minibatch * env nums,steps, action_emb)
            a_indices = a_indices.unsqueeze(3).expand(-1,-1,-1,self.action_e_dim).flatten(start_dim=0,end_dim=1) #  (  minibatch* args.num_envs,  num_steps,action_e_dim)

            a_epoch = torch.gather(input=a_epoch, index=a_indices, dim=1)[:,0:horizon,:] #  (  minibatch*  num_envs,  horizon,action_e_dim)
            o_out,o_hn = self.open_loop_rnn(a_epoch,chn_view) #  (  minibatch*  num_envs,  horizon, hiddens), _
            o_pred = self.predictor(o_out)  #  (  minibatch*  num_envs,  horizon, byol embedding size)

            # build open loop target
            with torch.no_grad():
                obs_o = obs_embs_t.reshape((steps,envNum,self.emb_dim)).unsqueeze(1).expand(-1,o_minibatch,-1,-1) # (steps, o_minibatch, env nums, obs emb_dim)
                obs_o = obs_o.reshape((obs_o.size(0), -1, self.emb_dim)).transpose(0,1)  # ( o_minibatch * env nums,steps, obs emb_dim)

                # get the indexes as before
                ascending_li += 1 #( num_steps,) get the next obs
                t_indices = indices + ascending_li.unsqueeze(0).unsqueeze(0).expand(o_minibatch, args.num_envs,
                                                                                    -1)  # (  minibatch, args.num_envs,  num_steps)
                t_indices = t_indices.unsqueeze(3).expand(-1, -1, -1, self.emb_dim).flatten(start_dim=0,end_dim=1)  # (  minibatch* num_envs,  num_steps, emb_dim)

                t_epoch = torch.gather(input=obs_o, index=t_indices, dim=1)[:, 0:horizon,:]  # (  minibatch*  num_envs,  horizon, emb_dim)
            loss_o = F.mse_loss(o_pred, t_epoch.detach(),reduction='mean')

            loss = loss_o + loss_c
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(loss)



def test_training():
    steps = 128
    horizon = 10
    o_minibatch = 3
    env = 128
    nums_layers = 2
    hidden = 64
    obs_dim = 201
    action_dim = 16
    action_e_dim = 30
    embed_dim = 150
    device = "cpu"
    dim_x, dim_y = 100,123
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.byol_open_loop_gru_max_horizon = horizon
    args.num_steps = steps
    args.byol_epochs = 100
    args.num_envs = env

    byol_encoder = BYOLEncoder(in_channels=1, out_size= 616, emb_dim=obs_dim)
    obs = torch.randn((args.num_steps, args.num_envs) + (dim_x, dim_y)).to(device)  # (steps, env nums, dimx,dimy)
    b_obs = obs.reshape((-1,) + (dim_x, dim_y))
    b = BYOL(obs_dim, action_dim, num_hidden=3, num_units=512,emb_dim=embed_dim, hidden_size =hidden,
             num_layers=nums_layers,action_e_dim=action_e_dim)
    actions = torch.randint(high = action_dim, size=(args.num_steps, args.num_envs))
    b_actions = actions.reshape(-1)
    c_hn_t = torch.randn((steps,nums_layers, env,hidden))  # closed_loop_hn_for_training (steps, byol num layers, env nums, byol hidden size)
    last_ep_c_hn = torch.randn((nums_layers,env,hidden))  # ( num_layers, num_envs ,  hidden_size),
    b.train_byol(args, device,
                   b_obs, b_actions, byol_encoder,
                   steps = args.num_steps,
                   c_hn_t = c_hn_t,
                   last_ep_c_hn=last_ep_c_hn,
                   o_minibatch =o_minibatch,
                   )

def test_slicing():
    #  steps * o_minibatch * env nums , num layers, hidden size)
    steps=8
    horizon =3
    o_minibatch =2
    env =6
    nums_layers=4
    hidden =5

    train_start = torch.randint(low=0, high=steps - horizon - 1, size=(o_minibatch, env),
                                dtype=torch.uint8)  # ( minibatch, num_envs)
    chn = torch.arange(0, end=steps*o_minibatch*env*nums_layers*hidden).reshape((steps*o_minibatch*env,nums_layers,hidden))
    print("train_start",train_start.size())
    chn_start = train_start.reshape(-1) + torch.arange(start=0,end=steps * o_minibatch * env ,
                                                       step=steps)  # (  minibatch * num_envs)
    chn_view = chn[chn_start, :, :].reshape((o_minibatch * env ,  nums_layers,  hidden)).transpose(0,1)  # ( num_layers, o_minibatch*envNum, hidden_size)
    print("chn_view.size()",chn_view.size())
    print("chn_view",chn_view)


def test_byol():
    byol_open_loop_gru_max_horizon = 100
    b = BYOL(5, 6, 7, 8, 9, 10)
    obs = torch.randn((3, 5))
    obs2 = torch.randn((3, 5))
    actions = torch.randint(low=0, high=6, size=(3,))

    p = b.get_action_predictions(obs)
    o_hiddens, intrinsic_loss, c_hn = b.get_intrinsic_reward(obs, actions, obs2)
    for step in range(99000):
        obs = torch.randn((3, 5))
        obs2 = torch.randn((3, 5))
        actions = torch.randint(low=0, high=6, size=(3,))
        print(o_hiddens.size(), "ll")
        o_pred, _ = b.get_action_predictions(obs, previous_c_hn=c_hn, o_hiddens=o_hiddens)
        o_hiddens, intrinsic_loss, c_hn = b.get_intrinsic_reward(obs, actions, obs2, c_hn, o_hiddens)

        if o_hiddens.size(
                1) > byol_open_loop_gru_max_horizon:  # (numlayers, previous states nums P+1 * envNum, hidden size H)
            if step % byol_open_loop_gru_max_horizon == 1:
                o_hiddens = o_hiddens[:, -byol_open_loop_gru_max_horizon:, :]
                o_hiddens_2 = o_hiddens.detach().clone()  # slicing is a view, how to save memory?
                del o_hiddens
                o_hiddens = o_hiddens_2
            else:
                o_hiddens = o_hiddens[:, -byol_open_loop_gru_max_horizon:, :]
        print(o_hiddens.size())

if __name__ == "__main__":

    test_training()
    # test_byol()
