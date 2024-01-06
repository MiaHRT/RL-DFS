import numpy as np
import cv2
import random
from Agent import Agent
from Environment import Maze

random.seed(1000)
np.random.seed(1000)

'''load and set maze and agent images'''
img = cv2.imread("./image/fig.jpg", 1)
maze = img.copy()
maze[400:408,554:562] = 0
maze = maze[48:440, 50:570, :]
agent = img[400:408,554:562].copy()
steplen = 8
boundary = (440-48,570-50)
maze_size = (49, 65)
destination = np.kron(np.ones((steplen,steplen,1), dtype=np.uint8),np.array([0,255,255], dtype=np.uint8))
trail0 = np.zeros((steplen,steplen,1), dtype=np.uint8)
trail0[3:5,3:5,:] = 1
trail0 = np.kron(trail0,np.array([0,255,0], dtype=np.uint8))
trail1 = np.ones((steplen,steplen,1), dtype=np.uint8)
trail1 = np.kron(trail1,np.array([0,255,0], dtype=np.uint8))
# calculate the wall position
wall = np.zeros(maze_size, dtype=np.uint8)
for i in range(maze_size[0]):
    for j in range(maze_size[1]):
        itmp = i*steplen
        jtmp = j*steplen
        if np.mean(maze[itmp:itmp+steplen, jtmp:jtmp+steplen, :]) > 80:
            wall[i,j] = 1
# cv2.imshow('maze',maze)
# cv2.waitKey()
# cv2.imshow('agent',agent)
# cv2.waitKey()
env = Maze(wall)

def project(maze, agent, i, j, dest=False):
    '''project the agent on the maze'''
    m = np.zeros((maze_size[0],maze_size[1],1), dtype=np.uint8)
    m[i,j,:] = 1
    maze_tmp = maze.copy()
    if agent is None:
        agent = np.kron(np.ones((steplen,steplen,1), dtype=np.uint8),np.array([255,255,255], dtype=np.uint8))
        tmp = np.kron(m,agent)
        maze_tmp[tmp!=0] = 0
    else:
        tmp = np.kron(m,agent)
        maze_tmp[tmp!=0] = 0
        maze_tmp = maze_tmp + tmp
        if dest:
            if np.mean(maze[tmp!=0])>10:
                return maze_tmp, True
            else:
                return maze_tmp, False

    return maze_tmp

def mouse_callbacks(event, x, y, flags, parm):
    '''mouse event callbacks: set the destination'''
    global have_dest, maze, spirit, di, dj, have_trail
    if not have_dest and event == cv2.EVENT_LBUTTONDOWN:
        # print(x,y)
        dest = (y,x)
        di = dest[0]//steplen
        dj = dest[1]//steplen
        if env.can_go(di, dj):
            maze, have_trail = project(maze, destination, di, dj, dest=True)
            have_dest = True
            env.set_dest_pos((di,dj))


'''initialzation'''
cv2.namedWindow('maze',cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('maze',mouse_callbacks)

# starting position of the agent
start_pos = (44, 63)
# start_pos = (25, 33)
# start_pos = (47, 35)
i, j = start_pos
print("PRESS [Esc]: shut the window.")
print("      [Space]: reset destination.")
print("CLICK any position to set destination.")
have_dest = False
cannot_reach = False
have_trail = False

'''set temper and iter of the agent'''
spirit = Agent(i,j,env,temper=10,iter=1)

env.add_agent(spirit)
have_shortest_path = False
maze = project(maze, trail1, i, j)


'''main thread'''
while True:
    if have_shortest_path:
        maze = project(maze, trail0, i, j)
        shortest_path_maze = project(shortest_path_maze, trail1, i, j)
        current_maze = project(shortest_path_maze, agent, i, j)
    else:
        maze = project(maze, trail0, i, j)
        current_maze = project(maze, agent, i, j)
    cv2.imshow('maze',current_maze)
    k = cv2.waitKey(10) # wait 10ms

    if have_dest and not cannot_reach:
        a = spirit.selection()
        if not have_shortest_path and spirit.shortest_path>0 and (i,j)==start_pos:
            shortest_path_maze = maze.copy()
            have_shortest_path = True
        if not have_shortest_path and a is None:
            i, j = start_pos
            spirit.set_pos(i,j)
        elif type(a)==tuple and not have_shortest_path:
            print("Can not reach the destination!")
            cannot_reach = True
        elif a is not None:
            i, j = spirit.update_pos(a)

    # ESC: shut the window
    if k == 27:
        # spirit.save_vertex()
        break
    # Space: reset
    elif k == 32:
        if (i==di and j==dj) or have_trail:
            maze = project(maze, None, di, dj)
            maze = project(maze, trail0, di, dj)
            have_trail = False
        else:
            maze = project(maze, None, di, dj)
        i, j = start_pos
        spirit.set_pos(i,j)
        have_shortest_path = False
        have_dest = False
        cannot_reach = False
