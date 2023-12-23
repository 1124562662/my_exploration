


class Buffer_add_helper:
    def __init__(self, traj_len_times, roll_out_len, buffer_size, n_envs):
        self.traj_top = 0
        self.top = 0

        # self.traj_add_count = 0
        self.traj_len_times = traj_len_times
        self.roll_out_len = roll_out_len
        self.buffer_size = buffer_size
        self.n_envs = n_envs

        self.die = False

    def get_indices_and_update(self, n_envs):
        add_envs = min(n_envs, self.buffer_size - self.top)
        q, w, e, r = self.top, self.top + add_envs,self.traj_top*self.roll_out_len, (self.traj_top + 1 )* self.roll_out_len
        self._update_indices(add_envs)
        return add_envs, q, w, e, r

    def _update_indices(self,add_envs):
        # print(self.top, "top")
        self.traj_top += 1
        self.traj_top %= self.traj_len_times
        if self.traj_top == 0:
            self.top += add_envs


    def no_add_cache(self,):
        return self.top + self.n_envs < self.buffer_size

    def adding_cache(self):
        return self.buffer_size - self.n_envs <= self.top < self.buffer_size

    def less_than_full(self,):
        return self.top < self.buffer_size

    def is_full(self, ):
        return self.top >= self.buffer_size
    #
    # def is_full(self, ):
    #     added = self.top * self.traj_len_times + self.traj_top
    #     total = self.buffer_size * self.traj_len_times
    #     self.die = True
    #     return added == total
