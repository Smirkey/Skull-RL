import numpy as np
from random import *
from inference_agents import *
from env import Env
import pygame
players = [Inf_bot('Al_Boubacar_son_of_Allah_Dessailly_5145'), Inf_bot('Al_Boubacar_son_of_Allah_Fijolet_7854'),
           Inf_bot('Antoine_Harstraxx_150'), Inf_human()]
players_names = {player.name:player for player in players}

def forward(state, agent_name, forward_pass = 1):
    return players_names[agent_name].get_probas(state, forward_pass)

def reward(amount, action, name):
    players_names[name].reward(amount, action)

if __name__ == '__main__':
    env = Env([player.name for player in players])
    env.step(1, forward, reward)
