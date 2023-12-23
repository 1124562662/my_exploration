"""in deterministic envs,
   If all the path a path following state s, has q(s,a) < tau, then s is explored"""

# 有些状态需要很多次经历，比如开始S_0,有些只需要很少的经历
# S0的ubc net 可能早早就 收敛到0，frontier上的ubc net才是关键

