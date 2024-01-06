class Maze(object):
    '''
    
    Functions:
        add_agent(): add a new agent to the environment
        can_go(): detect whether can go forward
        set_dest_pos(): set the position of the destination
        check_dest_pos(): check whether the current position is the destination
    '''
    def __init__(self, wall):
        self.__wall = wall
        self.__dest_pos = None

    def add_agent(self, agent):
        self.__agent = agent

    def can_go(self, i, j):
        if i<0 or j<0 or i>=self.__wall.shape[0] or j>=self.__wall.shape[1]:
            return False
        elif self.__wall[i,j]==1:
            return False
        else:
            return True
    
    def set_dest_pos(self, dest_pos):
        self.__dest_pos = dest_pos
        self.__agent.set_dest_pos(dest_pos)

    def check_dest_pos(self,i,j):
        if i==self.__dest_pos[0] and j==self.__dest_pos[1]:
            return True
        else:
            return False