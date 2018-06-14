from agent import Agent
from agent import Agent_Process
#from agent import copy
#from agent import breed
from env import Env_Process
from multiprocessing import *
from names import *
import numpy as np
from math import *
import random
import time
import tensorflow as tf
from time import gmtime, strftime

def connection_proxy(public_emit, private_recv, child_conn, index):
    while 1:
        data = child_conn.recv()
        key = data[0]
        public_emit.send([index, data])
        if key == 1:
            data = private_recv.recv()
            child_conn.send(data)


class Player:

    def __init__(self, name, process, conn, prestige, elo_base=1600):
        self.name = name
        self.elo = elo_base
        self.process = process
        self.conn = conn
        self.prestige = prestige


class EloModel:

    def __init__(self, players_per_game=4, model_class=Agent, players_per_gen=4, elo_base=1600,
                 elo_range=1400, K_factor=40, new_players_per_gen=2, games_per_gen=50, elo_graph = False):
        self.model_class = model_class
        #print("hello")
        self.players_per_gen = players_per_gen
        self.elo_base = elo_base
        self.players_per_game = players_per_game
        self.players_per_game = players_per_game
        self.current_gen = 0
        self.elo_range = elo_range
        self.elo_step = int(elo_range * 2 / 100)
        self.K_factor = K_factor
        self.new_players_per_gen = new_players_per_gen
        self.games_per_gen = games_per_gen
        self.current_players = []
        self.elo_graph = elo_graph
        #self.sess = tf.Session()
        #self.merged = tf.summary.merge_all()
        #self.train_writer = tf.summary.FileWriter('/logsd/', self.sess.graph)
        for x in range(self.players_per_gen):
            self.create_agent_process()

    def create_agent_process(self, father_name = None, mother_name=None, t_=0, epsilon=0.2, is_a_child = False):
        agent_parent_conn, agent_child_conn = Pipe()
        process = Agent_Process(agent_child_conn, players_per_game=self.players_per_game,
                                model_class=self.model_class,
                                elo_base=self.elo_base, players_per_gen=self.players_per_gen, is_a_child = is_a_child, father_name=father_name, mother_name=mother_name, t_=t_, epsilon=epsilon)
        agent_parent_conn.send([Ask_name, []])
        new_name = agent_parent_conn.recv()
        self.current_players.append(
            Player(new_name, process, agent_parent_conn, 0, self.elo_base))

    def get_elo_ranking(self):
        """self.current_players.sort(key=lambda c: c.elo)
        for player in self.current_players:
            tf.summary.scalar('elos', np.asarray(
                player.elo), family=player.name)

        self.summary = self.sess.run(self.merged)
        self.train_writer.add_summary(self.summary)
"""
        rank = 0
        self.current_players.sort(key=lambda c:c.elo)
        for x in self.current_players:
            print(rank, ': ', x.name, ' ', x.elo, ' prestige: ', x.prestige)
            rank += 1
        ranking = {player.name: player.elo for player in self.current_players}
        return ranking

    def eval_perf(self, player_elo, average_opponents_elo, perf):
        diff_elo = player_elo - average_opponents_elo
        diff_proba = diff_elo / self.elo_step / 100
        proba_winner = 0.5 + diff_proba
        gain = self.K_factor * (perf - proba_winner)
        return gain

    def eval_players(self, results, pool):
        player_names_dic = {
            player.name: player for player in self.current_players}
        pool = [player_names_dic[p] for p in pool]
        for result in results:
            if 1 in result:
                winner = pool[result.index(1)]
                losers = [player for player in pool]
                losers.remove(winner)
                total_losers_elo = sum([loser.elo for loser in losers])
                average_losers_elo =  total_losers_elo/ len(losers)
                gain = self.eval_perf(winner.elo, average_losers_elo, 1)
                winner.elo += gain
                for loser in losers:
                    weight = loser.elo / total_losers_elo
                    loser.elo -= gain * weight
            else:
                for player in pool:
                    opponents = [p for p in pool]
                    opponents.remove(player)
                    average_opponents_elo = sum(
                        [o.elo for o in opponents]) / len(opponents)
                    gain = self.eval_perf(
                        player.elo, average_opponents_elo, 0.5)
                    player.elo += gain
            for player in pool:
                player.elo = round(player.elo)

    def select_parents(self, n):
        self.current_players.sort(key=lambda c: c.elo)
        ranking = self.current_players
        #ranking.reverse()
        return ranking[:n]

    def get_highest_elo(self):

        self.current_players.sort(key=lambda c: c.elo)
        return self.current_players[-1].elo

    def format_elo(self):
        for player in self.current_players:
            player.elo = self.elo_base

    def reproduce(self, father, mother):
        #sess = tf.InteractiveSession()
        father.conn.send([Save, 1])
        conf = father.conn.recv()
        #father.process.terminate()
        #self.current_players.remove(father)
        mother.conn.send([Save, 1])
        conf = mother.conn.recv()
        #mother.process.terminate()
        #self.current_players.remove(mother)
        #father_copy = self.model_class(sess=sess, elo_base=self.elo_base, players_per_game=self.players_per_game, justforsex = True, original_name = father.name)
        #father_copy.sess.close()
        #print('Father: ', father.name, ' Mother: ', mother.name)
        #mother_copy = self.model_class(elo_base=self.elo_base, players_per_game=self.players_per_game,
        #                                                         justforsex = True, original_name = mother.name)
        #mother_copy.sess.close()
        #print('MOTHER CREATED')
        #new_agent = breed(sess, father_copy, mother_copy)
        #new_agent = copy(sess, father_copy)
        #new_name = new_agent.name
        #new_agent.save()
        #self.create_agent_process(original_name = father.name, is_a_child = True)
        self.create_agent_process(t_=conf[2],mother_name=mother.name, father_name = father.name, is_a_child=True, epsilon=conf[3])
        #self.create_agent_process(orinigal_name = mother.name, is_a_child =True)
        #del father_copy
        #del mother_copy
        #del new_agent

    def next_gen(self):
        print('NEXT GEN')
        self.current_players.sort(key=lambda p: p.elo)
        self.current_players.reverse()
        for player in self.current_players[-self.new_players_per_gen:]:
            player.process.terminate()
            self.current_players.remove(player)
        self.format_elo()
        for player in self.current_players:
        	player.prestige += 1
        
        parents = self.select_parents(self.new_players_per_gen)
        
        for i in range(len(parents)):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            while parent1 == parent2: parent2 = random.choice(parents)
            self.reproduce(parent1, parent2)

        del parents
        print('NEXT GEN ALIVE')

    def create_player_pools(self):
        players = self.current_players
        #players.sort(key=lambda c: c.elo)
        #players.reverse()
        #if not random.randrange(2):
        random.shuffle(players)
        pool = []
        pools = []
        for player in players:
            pool.append(player)
            if len(pool) == self.players_per_game:
                pools.append(pool)
                pool = []
                 
        return pools

    def receive_connections(self, connections):
        player_dic = {player.name: player for player in self.current_players}
        for connection in connections:
            data = connection.recv()
            for name in data.keys():
                player_dic[name].elo = data[name]

    def connection_handler(self, data, conns):
        conn = conns[data[0]]
        data = data[1]
        key = data[0]
        data = data[1]
        if key == Results:
            self.eval_players(data[0], data[1])
            return 0
        else:
            print(key, data)
        return 1

    def connections_manager(self, pipes, processes):
        public_recv, public_emit = Pipe()
        private_emits = []
        proxies = []
        for x in range(len(pipes)):
            private_emit, private_recv = Pipe()
            private_emits.append(private_emit)
            p = Process(target=connection_proxy, args=(
                public_emit, private_recv, pipes[x], x))
            proxies.append(p)
            p.start()
        print('Process Management starting')
        terminated = []
        while len(proxies) > len(terminated):
            data = public_recv.recv()
            if not self.connection_handler(data, private_emits):
                proxies[data[0]].terminate()
                terminated.append(proxies[data[0]])
                print(len(proxies), len(terminated))
        print('ALL PROCESSES CLOSED')
        for p in processes:
            p.terminate()
            del p
            

    def run(self):
        print('START TRAIN, MAX ELO: 3000')
        while self.get_highest_elo() < 3000:
            for x in range(self.games_per_gen):
                pools = self.create_player_pools()


                connections = []
                processes = []
                print(strftime("%Y-%m-%d %H:%M:%S ", gmtime()), str(x), '/', str(self.games_per_gen))
                for pool in pools:
                    print("------------------")
                    for player in pool:
                    	print(player.name, ' ', player.elo)
                    parent_conn, child_conn = Pipe()
                    connections.append(parent_conn)
                    processes.append(Env_Process(
                        child_conn, {p.name: p.conn for p in pool}))
                self.connections_manager(connections, processes)
                self.get_elo_ranking()
            self.next_gen()


"""def worker(meta_model, models):
    env = meta_model.Env(models)
    results = env.step(5)
    meta_model.eval_players(results,models)"""


"""class EloModel(object):
    
    def __init__(self, Env, players_per_game = 4, model_class = Agent, players_per_gen = 4, elo_base = 1600,
                 elo_range = 1400, K_factor = 40, new_players_per_gen = 8, games_per_gen = 1):
        self.Env = Env
        self.players_per_game = players_per_game
        self.current_gen = 0
        self.model_class = model_class
        self.players_per_gen = players_per_gen
        self.elo_base = elo_base
        self.current_players = self.gen_start_players()
        self.elo_range = elo_range
        self.elo_step = int(elo_range * 2 /100)
        self.K_factor = K_factor
        self.new_players_per_gen = new_players_per_gen
        self.games_per_gen = games_per_gen
        self.running_envs = []
        BaseManager.register('EloModel', self)
        self.manager = BaseManager()
        self.nbr_pools = 0 

    def gen_start_players(self):
        return [self.model_class(elo_base = self.elo_base, players_per_game = self.players_per_game)
                for x in range(self.players_per_gen)]

    def get_elo_ranking(self):
        self.current_players.sort(key = lambda c: c.elo)
        #print([e.elo for e in self.current_players])
        ranking = { player.name: player.elo for player in self.current_players}
        #print(ranking)
        return ranking

    def eval_perf(self,player_elo, average_opponents_elo,perf):
        diff_elo = player_elo - average_opponents_elo
        diff_proba = diff_elo / self.elo_step / 100
        proba_winner = 0.5 + diff_proba
        gain = self.K_factor * (perf - proba_winner)
        return gain

    def eval_players(self,results, pool):
        for result in results:
            if 1 in result:
                winner = pool[result.index(1)]
                losers = [player for player in pool]
                losers.remove(winner)
                average_losers_elo = sum([loser.elo for loser in losers])/len(losers)
                gain = self.eval_perf(winner.elo, average_losers_elo,1)
                winner.elo += gain
                shared_loss = gain / len(losers)
                for loser in losers: loser.elo -= shared_loss
            else:
                for player in pool:
                    opponents = [p for p in pool]
                    opponents.remove(player)
                    average_opponents_elo = sum([o.elo for o in opponents])/len(opponents)
                    gain = self.eval_perf(player.elo, average_opponents_elo, 0.5)
                    player.elo += gain
            for player in pool: 
                player.elo = round(player.elo)
        
    def select_parents(self, numParents):
        elos = [agent.elo for agent in self.current_players]
        sumElo = sum(elos)
        probs = [elo/sumElo for elo in elos]
        cumsum = np.cumsum(probs)
        p_index = []
        for i in range(numParents):
            r = random.random()
            curr = elos[0]
            for j in range(len(cumsum)):
                if abs(r - cumsum[j]) < abs(r - curr):
                    curr = cumsum[j]

            p_index.append(curr)
        parents = [self.current_players[int(index)] for index in p_index]
        return parents
    
    def get_highest_elo(self):
        self.current_players.sort(key = lambda c: c.elo)
        return self.current_players[-1].elo

    def format_elo(self):
        for player in self.current_players: player.elo = self.elo_base

    def next_gen(self):
        self.current_players.sort(key = lambda p: p.elo)
        self.current_players.reverse()
        self.current_players = self.current_players[:-self.new_players_per_gen]
        self.format_elo()
        parents = self.select_parents(self.new_players_per_gen)
        for i in range(len(parents)):
            self.current_players.append(breed(parents[i], parents[i + 1]))
        

    def create_player_pools(self):
        players = self.current_players
        players.sort(key = lambda c: c.elo)
        players.reverse()
        pool = []
        pools = []
        for player in players:
            pool.append(player)
            if len(pool) == self.players_per_game:
                pools.append(pool)
                pool = []
        return pools
        
    def get_running_envs(self):
       for env in self.running_envs:
           if not env.is_alive():
               self.running_envs.remove(env)
       return self.running_envs

    def receive_connections(self, connections):
        player_dic = {player.name:player for player in self.current_players}
        for connection in connections:
            data = connection.recv()
            for name in data.keys():
                player_dic[name].elo = data[name]


    def run(self):
        #print('START TRAIN, MAX ELO: 3000')
        self.manager.start()
        while self.get_highest_elo() < 3000:
            for x in range(self.games_per_gen):
                pools = self.create_player_pools()
                #print("ok")
                connections = []
                for pool in pools:
                    BaseManager.register('POOL '+str(self.nbr_pools),pool)
                    self.nbr_pools += 1
                    new_thread = Process(target = worker, args=[self,pool])
                    self.running_envs.append(new_thread)
                    new_thread.start()
                while len(self.get_running_envs())>0:
                    time.sleep(0.002)
                
            #print(self.get_running_envs())
                                      
            self.get_elo_ranking()
            self.next_gen()

"""
