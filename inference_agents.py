from agent import Agent
import tensorflow as tf
import random
import os
import pygame
from threading import Thread
import numpy as np
import time
from pygame.locals import *
MODEL_PATH = 'C:/Users/Anko/Desktop/code/SkullRl/model'
MODEL_PATHS = {}
for subdir, dirs, files in os.walk(MODEL_PATH):
    for d in dirs:
        path = subdir.replace("""\ """,'/')+'/'+d
        MODEL_PATHS[d[:-2]] = path
pygame.init()
window = pygame.display.set_mode((700,800))
pygame.display.flip()
graphic_grid = [[0 for x in range(7)] for x in range(8)]
class Graphics(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.name = "Thread"+str(random.randrange(100))
        self.turn  = True
    def run(self):
        while self.turn:
            time.sleep(0.1)
            window.fill((255,255,255))
            if 1:
                for event in pygame.event.get():
                    print(event)
                pos = [0,0]
                for line in graphic_grid:
                    pos[0] = 0
                    for obj in line:
                        if obj != 0:
                            obj.show(pos)
                        pos[0] += 100
                    pos[1] += 100
            pygame.display.flip()
                            
    def clear(self):
        global graphic_grid
        graphic_grid = [[0 for x in range(7)] for x in range(8)]

graphic_thread = Graphics()
graphic_thread.start()

class Graphic_Object:

    def __init__(self, pos):
        self.pos = pos
        graphic_grid[pos[0]][pos[1]] = self

    def show(self, pos):
        window.blit(self.skin, pos)
        
class Card(Graphic_Object):

    def __init__(self, pos, value):
        if value == 1:
            self.skin = pygame.image.load('images/flower.jpg')
        elif value == 2:
            self.skin = pygame.image.load('images/skull.jpg')
        elif value == -1:
            self.skin = pygame.image.load('images/card.jpg')
        self.skin = pygame.transform.scale(self.skin,(100,100))
        Graphic_Object.__init__(self, pos)

class Token(Graphic_Object):

    def __init__(self, pos, value, color):
        Graphic_bjects.__init__(self, pos)
        self.skin = pygame.font.Font(None, 35).render(str(value),0,color)
        
        


class Inf_bot(Agent):

    def __init__(self, source = 'RANDOM'):
        if source == 'RANDOM':
            source = random.choice(list(MODEL_PATHS.keys()))
        Agent.__init__(self, force_name = source)
        source_path = MODEL_PATHS[source]
        saver = tf.train.Saver()
        print('Restore...')
        old_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='model_agent_{}'.format(self.name))[:14]
        vars_dic = {}
        for var_name, _ in tf.contrib.framework.list_variables(source_path):
            var = tf.contrib.framework.load_variable(source_path, var_name)
            vars_dic[var_name] = var
        for var in old_vars:
            var = vars_dic[var.name[:-2]]
            
            
        print('Agent restored')


class Inf_human:

    def __init__(self):
        self.name = 'Human Player'

    def get_probas(self, state, forward_pass):
        print(state)
        graphic_thread.clear()
        pos = [7,2]
        for flower in range(state['FLOWERS']):
            Card(pos,1)
            pos[1]+=1
        if state['SKULL']: Card(pos,2)
        pos = [6,2]
        for card in state['OWN_FRONT']:
            if card != 0: Card(pos,card)
            pos[1] += 1
        front_poses = []
        for x in range(4):
            front_poses.append([x+2,5])
        for x in range(4):
            front_poses.append([0,x+2])
        for x in range(4):
            front_poses.append([x+2,1])
        x = 0
        for front in state['FRONTS']:
            for i in range(4):
                if front[i] != 0:
                    Card(front_poses[x], front[i])
        time.sleep(3)
        return np.random.rand(22)
    
    def reward(self,x,y):
        pass
        
        
        
        
            
        
    
