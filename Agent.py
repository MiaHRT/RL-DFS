import numpy as np
import random

class Agent(object):
    '''
    Parameters:
        iter: the maximun iteration of the algorithm
        itercount: the number of iterations
        Q_UB: upper bound of Q function value when calculation the softmax
        temper: randomness of exploration, the temperature gamma
        pos: current position of the agent
        ind_dest: the index of the destination in the vertex set V
        back: back direction of the agent
        actions: step list of the agent
        dist: distance
        last_vertex: last vertex the agent goes by
        last_action: last action the agent took in the last vertex
        breadth_first_v: the destination vertex when switching to breadth-first search
        vertex: V of G(V,D) (list of vertices)
        uncertain_vertex: list of uncertain vertices (arm)
        distance: D of G(V,D) (dict of distance between two vertices)
        Q: Q function, Q(uncertain vertex)
        shortest_path: length of the shortest path
        env: maze
    Functions:
        turn_round(): return the reverse direction of the back
        set_dest_pos(): set the position of the destination
        set_pos(): set the current position of the agent
        update_pos(): update the position of the agent
        selection(): select the action in the current position
        update(): update the G(V,E,W) of the maze
        pre_Dijkstra(): find the nearest vertices of the destination
        Dijkstra(): find the shortest path according to G(V,E,W)
        Dijkstra(select_breadth_first_v=True): update the Q function and select the breadth_first_v
        policy(): update the breadth_first_v according to Q function
    '''
    def __init__(self, i, j, env, temper=10, iter=5):
        self.__iter = iter
        self.__itercount = 0
        self.__Q_UB = 700
        self.__temper = temper
        self.__pos = [i,j]
        self.__back = "down"
        self.__actions = {"up":(-1,0), "right":(0,1), "down":(1,0), "left":(0,-1)}
        self.__dist = 0
        self.__last_vertex = 0
        self.__last_action = "up"
        self.__breadth_first_v = None
        self.__vertex = [(i,j)]
        self.__uncertain_vertex = [0]
        self.__distance = {}
        self.__Q = {}
        self.shortest_path = 0
        self.__env = env

    def turn_round(self, back):
        if back == "down":
            return "up"
        elif back == "up":
            return "down"
        elif back == "right":
            return "left"
        elif back == "left":
            return "right"

    def set_dest_pos(self, dest_pos):
        self.pre_Dijkstra(dest_pos)
        self.shortest_path = 0
        self.__itercount = 0

    def set_pos(self,i,j):
        self.__itercount += 1
        self.__pos = [i,j]
        self.__back = "down"
        self.__dist = 0
        self.__last_vertex = 0
        self.__last_action = "up"
        self.__breadth_first_v = None
        self.__Q = {}
    
    def get_memory_size(self):
        return len(self.__vertex)
    
    def save_vertex(self):
        np.save("vertex.npy", self.__vertex)

    def update_pos(self, a):
        self.__Q = {}
        if self.__env.check_dest_pos(self.__pos[0], self.__pos[1]):
            self.__dist = 0
            return self.__pos
        if (self.__pos[0], self.__pos[1]) in self.__vertex:
            tmp = self.__vertex.index((self.__pos[0], self.__pos[1]))
            self.__dist = 0
            self.__last_vertex = tmp
            self.__last_action = a
        value = self.__actions[a]
        self.__pos[0] += value[0]
        self.__pos[1] += value[1]
        self.__back = self.turn_round(a)
        if not self.__env.can_go(self.__pos[0], self.__pos[1]):
            print("wrong!")
        return self.__pos
    
    def selection(self):
        self.__dist += 1
        i, j = self.__pos
        action = []
        for key, value in self.__actions.items():
            itmp, jtmp = i+value[0],j+value[1]
            if key != self.__back  and self.__env.can_go(itmp,jtmp):
                action.append(key)

        if (i, j) in self.__vertex:
            tmp = self.update(visited=True)
            if self.__itercount < self.__iter:
                a = self.Dijkstra(select_breadth_first_v=True)
            else:
                a = self.Dijkstra()
            if a is None:
                if self.__env.check_dest_pos(i,j):
                    A = None
                elif len(action)==0:
                    self.__uncertain_vertex.remove(tmp)
                    A = self.__back
                else:
                    A = action[0]
            elif type(a)==tuple:
                A = a[0]
            elif type(a[0])==tuple:
                if self.__breadth_first_v is None:
                    if self.__itercount < self.__iter:
                        self.__itercount = self.__iter
                        return self.selection()
                    else:
                        return ()
                elif tmp == self.__breadth_first_v:
                    # print(1)
                    if tmp == 0:
                        value = self.__actions[self.__back]
                        itmp, jtmp = i+value[0],j+value[1]
                        if self.__env.can_go(itmp,jtmp):
                            action.append(self.__back)
                    action_tmp = action.copy()
                    for atmp, _ in a:
                        if atmp in action_tmp:
                            action_tmp.remove(atmp)
                    if tmp in self.__uncertain_vertex and len(action_tmp)==1:
                        self.__uncertain_vertex.remove(tmp)
                    A = action_tmp[0]
                else:
                    # print(2)
                    A = a[-1][0]
            else:
                # print(3)
                self.shortest_path = len(a)
                A = a[0]
        else:
            if len(action)==1:
                A = action[0]
            elif len(action)==0:
                self.update()
                return self.selection()
            else:
                self.__uncertain_vertex.append(len(self.__vertex))
                self.update()
                A = action[np.random.randint(0,len(action))]
        
        return A

    def update(self, visited=False):
        if visited:
            tmp = self.__vertex.index((self.__pos[0], self.__pos[1]))
            if self.__last_vertex == tmp:
                return tmp
            elif (self.__last_vertex, tmp) in self.__distance and self.__dist>=self.__distance[(self.__last_vertex, tmp)][1]:
                return tmp
        else:
            tmp = len(self.__vertex)
            self.__vertex.append((self.__pos[0], self.__pos[1]))
        self.__distance[(self.__last_vertex, tmp)] = (self.__last_action, self.__dist)
        self.__distance[(tmp, self.__last_vertex)] = (self.__back, self.__dist)
        # print(self.__vertex)
        # print(self.__distance)
        return tmp

    def pre_Dijkstra(self, dest_pos):
        if dest_pos in self.__vertex:
            self.__ind_dest = self.__vertex.index(dest_pos)
            return
        else:
            self.__ind_dest = len(self.__vertex)
            self.__vertex.append(dest_pos)
        i, j = dest_pos
        action = []
        for key, value in self.__actions.items():
            itmp, jtmp = i+value[0],j+value[1]
            if self.__env.can_go(itmp,jtmp):
                action.append(key)
        if len(action)==1:
            back = [self.turn_round(action[0])]
            i = [i + self.__actions[action[0]][0]]
            j = [j + self.__actions[action[0]][1]]
            last_action = action
        elif len(action)==2:
            back = [self.turn_round(action[0]), self.turn_round(action[1])]
            i = [i + self.__actions[action[0]][0], i + self.__actions[action[1]][0]]
            j = [j + self.__actions[action[0]][1], j + self.__actions[action[1]][1]]
            last_action = action
        else:
            self.__uncertain_vertex.append(self.__ind_dest)
            return
        
        for k in range(len(back)):
            dist = 1
            last_a = last_action[k]
            while True:
                if (i[k],j[k]) in self.__vertex:
                    tmp = self.__vertex.index((i[k],j[k]))
                    break
                else:
                    action = []
                    for key, value in self.__actions.items():
                        itmp, jtmp = i[k]+value[0],j[k]+value[1]
                        if key != back[k] and self.__env.can_go(itmp,jtmp):
                            action.append(key)
                    if len(action)==1:
                        dist += 1
                        value = self.__actions[action[0]]
                        i[k] += value[0]
                        j[k] += value[1]
                        last_a = action[0]
                        back[k] = self.turn_round(action[0])
                    elif len(action)==0:
                        break
                    else:
                        tmp = len(self.__vertex)
                        self.__uncertain_vertex.append(tmp)
                        self.__vertex.append((i[k],j[k]))
                        break
            if len(action)!=0:
                self.__distance[(tmp, self.__ind_dest)] = (self.turn_round(last_a), dist)
                self.__distance[(self.__ind_dest, tmp)] = (last_action[k], dist)
                if tmp in self.__uncertain_vertex:
                    n_a, d = 0, []
                    for key, value in self.__actions.items():
                        itmp, jtmp = i[k]+value[0],j[k]+value[1]
                        if self.__env.can_go(itmp,jtmp):
                            n_a += 1
                    for key, value in self.__distance.items():
                        if key[0] == tmp and value[0] not in d:
                            d.append(value[0])
                    if len(d) == n_a:
                        self.__uncertain_vertex.remove(tmp)

    def Dijkstra(self, select_breadth_first_v=False):
        if self.__env.check_dest_pos(self.__pos[0], self.__pos[1]):
            return None
        
        tmp = len(self.__vertex)
        a = []
        ind_cur = self.__vertex.index((self.__pos[0], self.__pos[1]))
        S = [ind_cur]
        path = {i:[] for i in range(tmp)}
        dist = np.ones(tmp)*np.inf
        dist[ind_cur] = 0
        V = list(range(tmp))
        del V[ind_cur]
        for i in V:
            if (ind_cur,i) in self.__distance:
                dist[i] = self.__distance[(ind_cur,i)][1]
        while len(V) != 0:
            ind = np.argmin([dist[i] for i in V])
            vtmp = V[ind]
            del V[ind]
            if np.isinf(dist[vtmp]):
                break
            S.append(vtmp)
            path[vtmp].append(vtmp)
            if vtmp == self.__ind_dest:
                break
            for i in V:
                if (vtmp,i) in self.__distance:
                    tmp = dist[vtmp]+self.__distance[(vtmp,i)][1]
                    if tmp < dist[i]:
                        dist[i] = tmp
                        path[i] = path[vtmp].copy()
        # print(path, dist)
        if select_breadth_first_v:
            for v in S:
                tmp = len(path[v])
                if tmp == 1:
                    if self.__ind_dest == v:
                        return self.__distance[(ind_cur,v)]
                    else:
                        a.append(self.__distance[(ind_cur,v)])
                if v in self.__uncertain_vertex:
                    self.__Q[v] = -dist[v]
            if len(a)==0:
                return None
            else:
                if len(a)==1 or self.__breadth_first_v not in self.__uncertain_vertex:
                    self.policy()
                if self.__breadth_first_v is not None and ind_cur != self.__breadth_first_v and len(path[self.__breadth_first_v])>0:
                    a.append(self.__distance[(ind_cur,path[self.__breadth_first_v][0])])
                return a
        else:
            i = ind_cur
            for j in path[self.__ind_dest]:
                a.append(self.__distance[(i,j)][0])
                i = j
            return a
    
    def policy(self):
        if len(self.__Q) == 0:
            self.__breadth_first_v = None
            return
        v = []
        Q = []
        for key, value in self.__Q.items():
            v.append(key)
            Q.append(value)
        Q = np.array(Q)/self.__temper
        Q = Q-min(Q)
        Q[Q>self.__Q_UB] = self.__Q_UB
        exp_Q = np.exp(Q)
        p = exp_Q/np.sum(exp_Q)
        self.__breadth_first_v = random.choices(v, weights=p, k=1)[0]
