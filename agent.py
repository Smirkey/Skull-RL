import tensorflow as tf
import numpy as np
from collections import deque
from names import *
import random
from multiprocessing import Process, Pipe
import os
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def fully_connected(INPUT, num_outputs):
    return tf.contrib.layers.fully_connected(INPUT, num_outputs)


def dropout(inputs):
    return tf.layers.dropout(inputs)


def rename(sess, checkpoint_dir, replace_from, replace_to):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    vars = []
    if 1:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                new_name = new_name.replace(replace_from, replace_to)

            var = tf.Variable(var, name=new_name)
            vars.append(var)
        sess.run(tf.global_variables_initializer())
    return vars


class Agent:

    def __init__(self, mother_name=None, father_name=None, justforsex=False, family_name=None, INPUTS=34, ACTIONS=22, epsilon=1, final_epsilon=0.001, start_epsilon=0.2, t_=0, t=0, num_states=10, elo_base=1600, players_per_game=4, batch=128, GAMMA=0.99, force_name=False, is_a_child=False):

        self.sess = tf.InteractiveSession(config=config)
        self.BATCH = batch
        self.father_name = father_name
        self.mother_name = mother_name
        self.first_name = names[random.randrange(len(names))]
        self.index = random.randint(0, 10000)
        if family_name == None:
            family_name = family_names[random.randrange(len(family_names))]
        self.family_name = family_name
        if force_name == False:
            self.name = '{}_{}_{}'.format(
                self.first_name, self.family_name, self.index)
        else:
            self.name = force_name
        self.elo = elo_base
        self.ACTIONS = ACTIONS
        self.INPUTS = INPUTS
        self.t = t
        self.GAMMA = GAMMA
        self.num_states = num_states
        self.state = tf.placeholder(
            tf.float32, shape=[None, self.INPUTS * self.num_states], name='state_agent_{}'.format(self.name))
        self.logits = self.model()
        self.start_epsilon = start_epsilon
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.replay_mem = 10000
        self.observe = 500
        self.explore = 1000000 - t_
        self.D = deque()
        self.action = tf.placeholder("float", [None, self.ACTIONS])
        self.y = tf.placeholder("float", [None])
        self.readout_action = tf.reduce_sum(tf.multiply(
            self.logits, self.action), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.y - self.readout_action))
        self.trainer = tf.train.AdamOptimizer(1e-6).minimize(self.loss)
        self.terminal = bool
        self.buffer = []
        self.r = 0
        self.s_t = []
        self.s_t1 = []
        self.should_train = False
        self.saver = tf.train.Saver()
        self.vars = [var for var in tf.trainable_variables() if self.name in var.name]
        self.clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.vars] 
        self.sess.run(tf.global_variables_initializer())
        self.is_a_child = is_a_child
        if self.is_a_child:
            #self.name = original_name
            father_vars = rename(self.sess, os.path.join(
                "model/{}_0".format(self.father_name)), self.father_name, self.name)
            mother_vars = rename(self.sess, os.path.join(
                "model/{}_0".format(self.mother_name)), self.mother_name, self.name)

            print("model loaded for {}, enjoy ur life dude".format(self.name))

            old_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='model_agent_{}'.format(self.name))[:14]

            father_vars_name_dic = {}
            mother_vars_name_dic = {}
            for var in father_vars:
                key = list(l for l in var.name)
                x = -1
                for l in range(len(key)):
                    if key[x] == ':':
                        end_index = x
                    elif key[x] == '_':
                        start_index = x
                        break
                    x -= 1

                key = key[:start_index] + key[end_index:]
                key = ''.join(key)
                if key not in father_vars_name_dic.keys():
                    father_vars_name_dic[key] = var

            for var in mother_vars:
                key = list(l for l in var.name)
                x = -1
                for l in range(len(key)):
                    if key[x] == ':':
                        end_index = x
                    elif key[x] == '_':
                        start_index = x
                        break
                    x -= 1

                key = key[:start_index] + key[end_index:]
                key = ''.join(key)
                if key not in mother_vars_name_dic.keys():
                    mother_vars_name_dic[key] = var

            for var in old_vars:
                z = tf.zeros([1], dtype=tf.float32)
                r = tf.random_uniform(father_vars_name_dic[
                                      var.name].shape, 0.99, 1.01)

                if random.random() > 0.5:
                    v = father_vars_name_dic[var.name]
                    w = tf.multiply(r, v)
                    self.sess.run(var.assign(w))

                else:
                    v = mother_vars_name_dic[var.name]
                    w = tf.multiply(r, v)
                    self.sess.run(var.assign(w))



        self.first_round = True 
        self.sess.run(tf.global_variables_initializer())
        print("new Agent created " + self.name)

        #self.logits.eval(feed_dict={self.state: np.zeros((1, self.INPUTS * self.num_states))})

    def model(self, reuse=False):
        with tf.variable_scope('model_agent_{}'.format(self.name)) as scope:
            if reuse:
                scope.reuse_variables()

            fc1 = fully_connected(self.state, 512)
            d1 = dropout(fc1)

            fc2 = fully_connected(d1, 1024)
            d2 = dropout(fc2)

            fc3 = fully_connected(d2, 512)
            d3 = dropout(fc3)

            fc4 = fully_connected(d3, 1024)
            d4 = dropout(fc4)

            fc5 = fully_connected(d4, 512)
            d5 = dropout(fc5)

            self.logits = tf.layers.dense(d5, self.ACTIONS)

            return self.logits

    def update_epsilon(self):
        self.epsilon -= (self.start_epsilon -
                         self.final_epsilon) / self.explore

    def get_probas(self, state, forward_pass = True):

        if not self.first_round:

            self.s_t = np.append(state, self.s_t[:self.num_states - 1])

            self.s_t = np.reshape(self.s_t, (self.num_states, self.INPUTS))
            self.s_t1 = np.reshape(
                self.s_t, (1, self.INPUTS * self.num_states))

            if len(self.buffer) > 0:

                self.buffer.append(np.reshape(
                    self.s_t1, (self.INPUTS * self.num_states)))

                if self.r != 0:

                    self.terminal = True

                else:

                    self.terminal = False

                self.t += 1
                self.buffer.append(self.terminal)
                self.train_agent()
                self.D.append(self.buffer)
                self.buffer = []

                if len(self.D) > self.replay_mem:
                    self.D.popleft()

        # NEED => SET OF LEGAL ACTIONS
        if self.first_round:
            self.s_t = np.asarray([state for x in range(self.num_states)])
            self.s_t1 = np.reshape(
                self.s_t, (1, self.INPUTS * self.num_states))
            self.first_round = False
            

        self.buffer.append(np.reshape(
            self.s_t1, (self.INPUTS * self.num_states)))
        if forward_pass:
            out = self.logits.eval(feed_dict={self.state: self.s_t1})[0]
            if random.random() <= self.epsilon:
                out = np.asarray([random.random() for x in range(self.ACTIONS)])
                # random action => cette action doit forcément être légale
        else:
            out = 1

        if self.epsilon > self.final_epsilon and self.t > self.observe:

            self.update_epsilon()

        return out

    def reward(self, value, action):

        self.r = value
        self.buffer.append(np.eye(self.ACTIONS)[action])
        self.buffer.append(self.r)

    def save(self):
        if not os.path.exists('./model/{}_0'.format(self.name)):
            os.makedirs('./model/{}_0'.format(self.name))
        self.saver.save(self.sess, './model/{}_0/{}'.format(self.name, self.t))

    def train_agent(self):

        if self.t > self.observe:

            minibatch = random.sample(self.D, self.BATCH)
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]
            y_batch = []
            logits_j1_batch = self.logits.eval(
                feed_dict={self.state: s_j1_batch})  # pas sur

            for i in range(len(minibatch)):  # Ou sauvegarder sars' dans la replay mem ?

                terminal = minibatch[i][4]

                if terminal:

                    y_batch.append(r_batch[i])

                else:

                    y_batch.append(
                        r_batch[i] + self.GAMMA * np.max(logits_j1_batch[i]))

            self.sess.run([self.trainer],feed_dict={self.y: y_batch, self.action: a_batch, self.state: s_j_batch})
            self.sess.run(self.clip)
            self.should_train = False


"""
def breed(sess, parentA, parentB, randrange=0.05):

    newAgent = Agent(force_name = 'Temorary_Agent' + str(random.randrange(10000)))
    
    parentA_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model_agent_{}".format(parentA.name))[:14]
    parentB_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model_agent_{}".format(parentB.name))[:14]
    indeces = [0, 2, 4, 6, 8, 10, 12]
    scopes = {2: 1, 4: 2, 6: 3, 8: 4, 10: 5}

    for i in range(len(parentA_weights)):

        if i in indeces:

            r = tf.random_uniform(parentA_weights[i].shape, 1 - randrange, 1 + randrange)

            if i == 0:
                
                path = "model_agent_{}/fully_connected/weights:0".format(newAgent.name)
                path2 = "model_agent_{}/fully_connected/biases:0".format(newAgent.name)

            elif i == 12:

                path = "model_agent_{}/dense/kernel:0".format(newAgent.name)
                path2 = "model_agent_{}/dense/bias:0".format(newAgent.name)

            else:
                path = "model_agent_{}/fully_connected_{}/weights:0".format(newAgent.name, scopes[i])
                path2 = "model_agent_{}/fully_connected_{}/biases:0".format(newAgent.name, scopes[i])

            if random.random() > 0.5:
                print(path, path2, len(parentA_weights))
            
                w = tf.multiply(r, parentA_weights[i])
                sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=path)[0].assign(w))
                sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=path2)[0].assign(parentA_weights[i + 1]))
            
            else:
                print(path, path2, len(parentA_weights))
            
                w = tf.multiply(r, parentB_weights[i])
                sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=path)[0].assign(w))
                sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=path2)[0].assign(parentB_weights[i + 1]))
    
    return newAgent

def copy(sess, Parent, randrange=0.05):
    newAgent = Agent(force_name = 'Temorary_Agent' + str(random.randrange(10000)))
    parent_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model_agent_{}".format(Parent.name))[:14]
    indeces = [0, 2, 4, 6, 8, 10, 12]
    scopes = {2: 1, 4: 2, 6: 3, 8: 4, 10: 5}

    for i in range(len(parent_weights)):

        if i in indeces:

            r = tf.random_uniform(parent_weights[i].shape, 1 - randrange, 1 + randrange)
            if i == 0:
        	    path = "model_agent_{}/fully_connected/weights:0".format(newAgent.name)
        	    path2 = "model_agent_{}/fully_connected/biases:0".format(newAgent.name)
            elif i == 12:
        	    path = "model_agent_{}/dense/kernel:0".format(newAgent.name)
        	    path2 = "model_agent_{}/dense/bias:0".format(newAgent.name)
            else:
        	    path = "model_agent_{}/fully_connected_{}/weights:0".format(newAgent.name, scopes[i])
        	    path2 = "model_agent_{}/fully_connected_{}/biases:0".format(newAgent.name, scopes[i])


            w = tf.multiply(r, parent_weights[i])

            sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=path)[0].assign(w))
            sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=path2)[0].assign(parent_weights[i + 1]))


    return newAgent
"""


class Agent_Process(Process):

    def __init__(self, conn, players_per_game=4, model_class=Agent, elo_base=1600, players_per_gen=8,
                 force_name=False, is_a_child=False, father_name=None, mother_name=None, t_=0, epsilon=0.2):
        self.model_class = model_class
        self.players_per_gen = players_per_gen
        self.elo_base = elo_base
        self.players_per_game = players_per_game
        self.conn = conn
        self.force_name = force_name
        self.father_name = father_name
        self.mother_name = mother_name
        self.is_a_child = is_a_child
        self.t_ = t_
        self.epsilon = epsilon
        Process.__init__(self, target=self.step)
        self.start()

    def gen_model(self):
        return self.model_class(elo_base=self.elo_base, players_per_game=self.players_per_game,
                                force_name=self.force_name, is_a_child=self.is_a_child, father_name=self.father_name, mother_name=self.mother_name, t_=self.t_)

    def connection_handler(self, data):
        key = data[0]
        data = data[1:]
        if key == Get_probas:
            state = data[0][0]
            forward_pass = data[0][1]
            conn2 = data[1]

            probas = self.model.get_probas(state, forward_pass)
            if forward_pass: conn2.send(probas)
            if self.model.should_train:
                self.model.train_agent()
            else:
                self.model.should_train = True
        elif key == Reward:
            value = data[0][0]
            action = data[0][1]
            self.model.reward(value, action)
        elif key == Save:
            self.model.save()
            self.conn.send([Save, 1, self.model.t, self.model.epsilon])

    def refresh_player_dict(self):
        self.player_names_dic = {
            player.name: player for player in self.current_players}

    def step(self):
        #self.sess = tf.InteractiveSession()
        self.model = self.gen_model()
        self.conn.send(self.model.name)
        self.continue_process = True
        while self.continue_process:
            data = self.conn.recv()
            self.connection_handler(data)
