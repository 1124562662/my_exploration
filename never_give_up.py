import torch
import torch.nn.functional as F
from torch import nn, optim
from Networks import TensorLinear

def create_mlp(inp_size, num_hidden, num_units, out_size):
    layers = [nn.Linear(inp_size, num_units), nn.ReLU()]
    for _ in range(num_hidden - 1):
        layers.append(nn.Linear(num_units, num_units))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(num_units, out_size))
    return nn.Sequential(*layers)


def create_3d_mlp(inp_size, num_hidden, num_units, out_size,env_num):
    a = 0.1
    layers = [TensorLinear(channel=env_num, outd=num_units, ind=inp_size, a=a), nn.LeakyReLU(a)]
    for _ in range(num_hidden - 1):
        layers.append(TensorLinear(channel=env_num, outd=num_units, ind=num_units, a=a))
        layers.append(nn.LeakyReLU(a))
    layers.append(TensorLinear(channel=env_num, outd=out_size, ind=num_units, a=a))
    return nn.Sequential(*layers)


# count embedding of the Byol prediction, at action selection stage
# update RND at every step, and re-initialize the network at the start of every rollout
# re-initialize based on dones!
# 相比于 kernel-based methods的线性空间复杂度，这个只有 常数的复杂度
# TODO 只会actions之间横向对比，不会不同steps之间纵向对比，所以不用考虑不同steps之间scale不同的问题？？？
# TODO 加入 loss normalization
class Within_episode_BYOL_RND(nn.Module):
    def __init__(self, obs_embeds_dim,action_embeds_dim, action_num,env_num,
                 embed_dim=50, num_hidden=2, num_units=50, learning_rate=0.01,):
        super(Within_episode_BYOL_RND, self).__init__()
        self.obs_embeds_dim = obs_embeds_dim
        self.action_num = action_num
        self.action_embeds_dim = action_embeds_dim
        self.env_num = env_num

        self.rnd = create_3d_mlp(obs_embeds_dim + action_embeds_dim, num_hidden + 1, num_units, embed_dim,env_num)
        self.rnd_target = create_3d_mlp(obs_embeds_dim + action_embeds_dim, num_hidden, num_units, embed_dim,env_num)
        self.optimizer = optim.Adam(self.rnd.parameters(), lr=learning_rate)

    def get_similarity_and_update_RND(self,
                                      byol_action_embedder,
                                      obs_embeds, #  (horizon_batch, env num, action_pred_num , Byol state embedding embedding size)
                                      epoch:int = 2,
                                      ):
        actions = torch.arange(start=0, end=self.action_num)  # (action_pred_num,)
        horizon_batch = obs_embeds.size(0)
        action_embs = byol_action_embedder(actions).unsqueeze(0).expand(self.env_num, -1, -1).unsqueeze(0).expand(horizon_batch,-1, -1, -1)  # (horizon_batch, env_num, action_pred_num, action_emb)

        embs = torch.cat((obs_embeds,action_embs), dim=3).transpose(2,3).detach()  #  (horizon_batch, self.env_num,  obs_emb + action_emb, action_pred_num)
        print(embs.size())
        with torch.no_grad():
            et = self.rnd_target(embs) #  (horizon_batch,  env_num, embed_dim, action_pred_num)

        for step in range(epoch):
            e = self.rnd(embs) #  (horizon_batch, env_num, embed_dim, action_pred_num)
            loss = F.mse_loss(e,et.detach(),reduction='none') # (horizon_batch, env_num, embed_dim, action_pred_num)
            if step == 0:
                with torch.no_grad():
                    loss_return = loss.mean(2).mean(0).detach().clone()  # ( env_num , action_pred_num)
            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_return # ( env_num , action_pred_num)

    def re_initialize_network(self, dones_this_step):
        # dones_this_step    (envs,)
        for i in range(dones_this_step.size(0)):
            if dones_this_step[i] == 1:
                for t in self.rnd.children():
                    if t.__class__.__name__ == "TensorLinear":
                        t.reset_channel(i)
                        #print(t.__class__.__name__)
                for t in self.rnd_target.children():
                    if t.__class__.__name__ == "TensorLinear":
                        t.reset_channel(i)
                        #print(t.__class__.__name__)





def test_Within_episode_BYOL_RND():
    obs_embeds_dim, action_embeds_dim, action_num, env_num =7,6,5,40
    w= Within_episode_BYOL_RND(obs_embeds_dim, action_embeds_dim, action_num, env_num)
    obs_embeds = torch.randn((env_num,action_num, obs_embeds_dim))  # (env num, action_pred_num , Byol state embedding embedding size)
    byol_emeb = torch.nn.Embedding(num_embeddings=action_num, embedding_dim=action_embeds_dim)
    loss_return = w.get_similarity_and_update_RND(byol_action_embedder=byol_emeb,obs_embeds=obs_embeds)  # (self.env_num , action_pred_num)

    print(loss_return.size())
    dones =torch.zeros( (env_num,))
    dones[5]=1
    w.re_initialize_network(dones)
    dones = torch.randint(high =2,size=(env_num,))
    w.re_initialize_network(dones)


# directly count the (s,a) pair
# update RND at every step, and re-initialize the network at the start of every rollout
# re-initialize based on dones!
# 相比于 kernel-based methods的线性空间复杂度，这个只有 常数的复杂度
# TODO 加入 loss normalization
class Within_episode_SA_RND(nn.Module):
    def __init__(self, obs_embeds_dim, action_embeds_dim, action_num, env_num,
                 embed_dim=50, num_hidden=2, num_units=50, learning_rate=0.01, ):
        super(Within_episode_SA_RND, self).__init__()
        self.obs_embeds_dim = obs_embeds_dim
        self.action_num = action_num
        self.action_embeds_dim = action_embeds_dim
        self.env_num = env_num

        self.rnd = create_3d_mlp(obs_embeds_dim + action_embeds_dim, num_hidden + 1, num_units, embed_dim, env_num)
        self.rnd_target = create_3d_mlp(obs_embeds_dim + action_embeds_dim, num_hidden, num_units, embed_dim, env_num)
        self.optimizer = optim.Adam(self.rnd.parameters(), lr=learning_rate)

    def get_similarity_and_update_RND(self,
                                      byol_action_embedder,
                                      obs_embeds,# (horizon_batch, env num, Byol state embedding embedding size)
                                      epoch: int = 2,
                                      ):
        obs_embeds = obs_embeds.unsqueeze(2).expand(-1,-1,self.action_num,-1) # (horizon_batch, env num, action_pred_num , Byol state embedding embedding size)
        actions = torch.arange(start=0, end=self.action_num)  # (action_pred_num,)
        horizon_batch = obs_embeds.size(0)
        action_embs = byol_action_embedder(actions).unsqueeze(0).expand(self.env_num, -1, -1).unsqueeze(0).expand(
            horizon_batch, -1, -1, -1)  # (horizon_batch, env_num, action_pred_num, action_emb)

        embs = torch.cat((obs_embeds, action_embs), dim=3).transpose(2,
                                                                     3).detach()  # (horizon_batch, self.env_num,  obs_emb + action_emb, action_pred_num)
        print(embs.size())
        with torch.no_grad():
            et = self.rnd_target(embs)  # (horizon_batch,  env_num, embed_dim, action_pred_num)

        for step in range(epoch):
            e = self.rnd(embs)  # (horizon_batch, env_num, embed_dim, action_pred_num)
            loss = F.mse_loss(e, et.detach(), reduction='none')  # (horizon_batch, env_num, embed_dim, action_pred_num)
            if step == 0:
                with torch.no_grad():
                    loss_return = loss.mean(2).mean(0).detach().clone()  # ( env_num , action_pred_num)
            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_return  # ( env_num , action_pred_num)

    def re_initialize_network(self, dones_this_step):
        # dones_this_step    (envs,)
        for i in range(dones_this_step.size(0)):
            if dones_this_step[i] == 1:
                for t in self.rnd.children():
                    if t.__class__.__name__ == "TensorLinear":
                        t.reset_channel(i)
                        # print(t.__class__.__name__)
                for t in self.rnd_target.children():
                    if t.__class__.__name__ == "TensorLinear":
                        t.reset_channel(i)
                        # print(t.__class__.__name__)

def test_Within_episode_SA_RND():
    obs_embeds_dim, action_embeds_dim, action_num, env_num =7,6,5,40
    w= Within_episode_SA_RND(obs_embeds_dim, action_embeds_dim, action_num, env_num)
    obs_embeds = torch.randn((env_num, obs_embeds_dim))  # (env num, Byol state embedding embedding size)
    byol_emeb = torch.nn.Embedding(num_embeddings=action_num, embedding_dim=action_embeds_dim)
    loss_return = w.get_similarity_and_update_RND(byol_action_embedder=byol_emeb,obs_embeds=obs_embeds)  # (self.env_num , action_pred_num)

    print(loss_return.size())
    dones =torch.zeros( (env_num,))
    dones[5]=1
    w.re_initialize_network(dones)
    dones = torch.randint(high =2,size=(env_num,))
    w.re_initialize_network(dones)


# as in NGU
class Within_episode_kernel(nn.Module):
    pass
    # def __init__(self, ):
    #     super(Within_episode_kernel,self).__init__()
    #     self.episodic_memory = None  # tuples of ()
    #
    #     self.distance_mean = 0
    #     self.distance_count = 0
    #
    # def add(self, current_c_state: Tensor,):
    #     if self.episodic_memory is None:
    #         self.episodic_memory =  current_c_state.unsqueeze(1).numpy() # tuples of size 1
    #     else:
    #         pass
    #
    # def re_initialize_history(self, dones):# TODO -- torch multi-processing
    #     def compute_intrinsic_reward(
    #             episodic_memory: List,
    #             current_c_state: Tensor,
    #             k=10,
    #             kernel_cluster_distance=0.008,
    #             kernel_epsilon=0.0001,
    #             c=0.001,
    #             sm=8,
    #     ) -> float:
    #         state_dist = [(c_state, torch.dist(c_state, current_c_state)) for c_state in episodic_memory]
    #         state_dist.sort(key=lambda x: x[1])
    #         state_dist = state_dist[:k]
    #         dist = [d[1].item() for d in state_dist]
    #         dist = np.array(dist)
    #
    #         # TODO: moving average
    #         dist = dist / np.mean(dist)
    #
    #         dist = np.max(dist - kernel_cluster_distance, 0)
    #         kernel = kernel_epsilon / (dist + kernel_epsilon)
    #         s = np.sqrt(np.sum(kernel)) + c
    #
    #         if np.isnan(s) or s > sm:
    #             return 0
    #         return 1 / s
#
#
#
# def test_Within_episode_kernel():
#     w = Within_episode_kernel()
#     nstate = torch.randn((3, 5))
#     w.add(nstate)
#     for i in range(1, 2000):
#         nstate = torch.randn((3, 5))
#         w.add(nstate)
#         state = torch.randn((2 * i, 3, 5))
#         print(w.get_similarity_kernel(state).size())

if __name__ == "__main__":
    test_Within_episode_SA_RND()
    # test_Within_episode_BYOL_RND()
    # test_Within_episode_kernel()




# def get_similarity_kernel(
#         self,
#         current_state: Tensor,  # (byol_predic_horizon * action_dim =N , envNum, embedding)
#         k=8,
#         kernel_cluster_distance=0.008,
#         kernel_epsilon=0.0001,
#         c=0.001,
#         sm=8,
# ):
#     state_nums = self.episodic_memory.size(0)
#     N = current_state.size(0)
#     mul_current_states = current_state.unsqueeze(0).expand(state_nums, -1, -1,
#                                                            -1)  # ( state numbers,N,env nums,byol emb)
#     memory_view = self.episodic_memory.unsqueeze(1).expand(-1, N, -1, -1)  # ( state numbers,N,env nums,byol emb)
#     distances = torch.abs(mul_current_states - memory_view)  # ( state numbers,N,env nums,byol emb)
#     distances = distances.mean(-1)  # ( state numbers,N,env nums)
#     distances = distances.permute(2, 1, 0)  # ( env nums,N, state numbers)
#
#     # update  moving average of the mean distances
#     distances_mv = distances.mean().item()  # (1,)
#     distances_count = distances.numel()  # int
#     self.distance_mean = self.distance_mean + (distances_mv - self.distance_mean) / float(distances_count)
#     self.distance_count += int(distances_count / 2)  # bias to current distances
#
#     # for every env, every partial history's prediction, sort
#     sorted, indices = torch.sort(distances, dim=-1)  # ( env nums,N, state numbers)
#     cut = k if state_nums > k else state_nums
#     top_k = sorted[:, :, :cut]  # ( env nums,N, cut)
#     top_k /= torch.tensor(self.distance_mean)  # .to(top_k.get_device()) # ( env nums,N, cut)
#     top_k = torch.max(top_k - kernel_cluster_distance * torch.ones_like(top_k),
#                       torch.zeros_like(top_k))  # ( env nums,N, cut)
#     top_k = torch.div(kernel_epsilon * torch.ones_like(top_k),
#                       top_k + kernel_epsilon * torch.ones_like(top_k))  # ( env nums,N, cut)
#     top_k = top_k.mean(-1)  # ( env nums,N )
#     s = top_k + c * torch.ones_like(top_k)  # ( env nums,N )
#     s = torch.div(torch.ones_like(s), s)  # ( env nums,N )
#     s[torch.isnan(s)] = 0  # ( env nums,N )
#     s[s < float(1 / sm)] = 0  # ( env nums,N )
#
#     return s  # ( env nums,N )
#





