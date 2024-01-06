import numpy as np
import random
import matplotlib.pyplot as plt
from Agent import Agent
from Environment import Maze

random.seed(1000)
np.random.seed(1000)

'''set maze'''
img = plt.imread("./image/fig.jpg", 1)
maze = img.copy()
maze[400:408,554:562] = 0
maze = maze[48:440, 50:570, :]
steplen = 8
boundary = (440-48,570-50)
maze_size = (49, 65)
wall = np.zeros(maze_size)
for i in range(maze_size[0]):
    for j in range(maze_size[1]):
        itmp = i*steplen
        jtmp = j*steplen
        if np.mean(maze[itmp:itmp+steplen, jtmp:jtmp+steplen, :]) > 80:
            wall[i,j] = 1
env = Maze(wall)

'''load vertices'''
dest_pos = np.load("./vertex.npy")
dest_pos = np.delete(dest_pos, 0, 0)
'''plot the vertices'''
# vertex = np.zeros(maze_size)
# for di, dj in dest_pos:
#     vertex[di, dj] = 10
# wall_tmp = wall.copy()
# wall_tmp[wall_tmp==1] = -10
# wall_tmp[np.multiply(vertex==0, wall_tmp==0)] = -10/2.5
# plt.imshow(wall_tmp+vertex)
# plt.show()

'''plot steps and memory size'''
start_pos = (44, 63)
# start_pos = (25, 33)
# start_pos = (47, 35)
n_step = {}
n_memo = {}
n = 0
for di, dj in dest_pos:
    dest = (di, dj)
    n_step[dest] = []
    n_memo[dest] = []
    for _ in range(10):
        i, j = start_pos
        spirit = Agent(i,j,env,temper=10,iter=1)
        env.add_agent(spirit)
        env.set_dest_pos(dest)
        while True:
            a = spirit.selection()
            if a is None:
                # print("Reach the destination!")
                n_step[dest].append(n)
                n_memo[dest].append(spirit.get_memory_size())
                break
            elif type(a)==tuple:
                print("Can not reach the destination!")
                n_step[dest].append(np.nan)
                n_memo[dest].append(np.nan)
                break
            elif a is not None:
                spirit.update_pos(a)
                n += 1
        n = 0

# print(n_step)
# print(n_memo)
min_step = np.zeros(maze_size)
max_step = np.zeros(maze_size)
min_memo = np.zeros(maze_size)
max_memo = np.zeros(maze_size)
tmp = np.full(maze_size, False, dtype=bool)
for di, dj in dest_pos:
    dest = (di, dj)
    tmp[di, dj] = True
    min_step[di, dj] = min(n_step[dest])
    max_step[di, dj] = max(n_step[dest])
    min_memo[di, dj] = min(n_memo[dest])
    max_memo[di, dj] = max(n_memo[dest])

print("min min_step={}, median min_step={}, max min_step={}".format(np.min(min_step[tmp]),np.median(min_step[tmp]),np.max(min_step[tmp])))
print("min max_step={}, median max_step={}, max max_step={}".format(np.min(max_step[tmp]),np.median(max_step[tmp]),np.max(max_step[tmp])))
print("min min_memo={}, median min_memo={}, max min_memo={}".format(np.min(min_memo[tmp]),np.median(min_memo[tmp]),np.max(min_memo[tmp])))
print("min max_memo={}, median max_memo={}, max max_memo={}".format(np.min(max_memo[tmp]),np.median(max_memo[tmp]),np.max(max_memo[tmp])))

# fig, ax = plt.subplots(2,2,layout='tight')
# ax[0,0].hist(min_step[tmp])
# ax[0,0].set_title("min steps (median={:d})".format(int(np.median(min_step[tmp]))))
# ax[0,0].grid()
# ax[0,1].hist(max_step[tmp])
# ax[0,1].set_title("max steps (median={:d})".format(int(np.median(max_step[tmp]))))
# ax[0,1].grid()
# ax[1,0].hist(min_memo[tmp])
# ax[1,0].set_title("min memory size (median={:d})".format(int(np.median(min_memo[tmp]))))
# ax[1,0].grid()
# ax[1,1].hist(max_memo[tmp])
# ax[1,1].set_title("max memory size (median={:d})".format(int(np.median(max_memo[tmp]))))
# ax[1,1].grid()
# plt.show()

# wall_tmp = wall.copy()
# wall_tmp[wall_tmp==1] = -np.max(min_step)
# wall_tmp[np.multiply(min_step==0, wall_tmp==0)] = -np.max(min_step)/2.5
# plt.imshow(wall_tmp+min_step)
# plt.show()
# wall_tmp = wall.copy()
# wall_tmp[wall_tmp==1] = -np.max(max_step)
# wall_tmp[np.multiply(max_step==0, wall_tmp==0)] = -np.max(max_step)/2.5
# plt.imshow(wall_tmp+max_step)
# plt.show()
# wall_tmp = wall.copy()
# wall_tmp[wall_tmp==1] = -np.max(min_memo)
# wall_tmp[np.multiply(min_memo==0, wall_tmp==0)] = -np.max(min_memo)/2.5
# plt.imshow(wall_tmp+min_memo)
# plt.show()
# wall_tmp = wall.copy()
# wall_tmp[wall_tmp==1] = -np.max(max_memo)
# wall_tmp[np.multiply(max_memo==0, wall_tmp==0)] = -np.max(max_memo)/2.5
# plt.imshow(wall_tmp+max_memo)
# plt.show()