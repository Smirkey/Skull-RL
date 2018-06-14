import numpy as np
from random import *
from multiprocessing import Process, Pipe
from names import *
import math
names = ['Mickael','Thierry','Georges','Antoine']

def format_state(state):
        del state['HAS_HIDDEN']
        del state['NBR_CARDS']
        del state['BET'] 
        new_state = []
        for v in state.values():
            try:
                for x in v:
                    try: new_state.append(float(x))
                    except : 
                         for y in x: new_state.append(float(y))
            except: new_state.append(float(v))
        state = np.asarray(new_state)
        return state

class Card:
    dic = {-1:'EMPY',0:'HIDDEN',1:'FLOWER',2:'SKULL'}
    def __init__(self,value):
        self.state = -1
        self.value = value
    def __str__(self, public = 1):
        if public:
            return Card.dic[self.state]
        else:
            return Card.dic[self.value]
actions = []
class Action:
    def __init__(self,typ,value,index):
        self.type = typ 
        self.value = value
        self.index = index
        actions.append(self)
    def gen_actions():
        index = 0
        for x in range(2):
            Action('CARD',x,index)
            index += 1
        for x in range(16):
            Action('BET',x,index)
            index += 1
        Action('ABANDON',0,index)
        index += 1
        for x in range(3):
            Action('SHOW',x,index)
            index += 1

Action.gen_actions()
        
        
class Player:
    
    index = 0
    
    def reset(self):
        self.pocket = [Card(1),Card(1),Card(1),Card(2)]
        self.frontCards = []
        self.points = 0
        self.relative_order = 0
        self.reward_pending = False
        self.last_skull = 0
        self.max_bet = 0
        
    def __init__(self, name):
        if Player.index > 3:
            Player.index = 0
        self.index = Player.index
        Player.index += 1
        self.name = name
        
    def public_front(self):
        front = []
        for x in range(4):
            try:
                front.append(self.frontCards[x].state)
            except:
                front.append(0)
        return front
    
    def private_front(self):
        front = []
        for x in range(4):
            try:
                front.append(self.frontCards[x].value)
            except:
                front.append(0)
        return front
    
    def clear(self):
        for card in self.frontCards:
            self.pocket.append(card)
        self.frontCards = []
        self.refresh_stats()
        
    def has_skull(self):
        if 2 in [card.value for card in self.pocket]:return 1
        else: return 0
    
    def nbr_flowers(self):
        return [card.value for card in self.pocket].count(1)
    
    def play_flower(self):
        for card in self.pocket:
            if card.value == 1:
                card.state = -1
                self.frontCards.append(card)
                self.pocket.remove(card)
                return
            
    def play_skull(self):
        for card in self.pocket:
            if card.value == 2:
                card.state = -1
                self.frontCards.append(card)
                self.pocket.remove(card)
                return
        
    def show(self):
        i = 1
        while i <= len(self.frontCards):
            card = self.frontCards[-i]
            if card.state == -1:
                card.state = card.value
                return card.state
            i+=1
        return 0
    
    def pocket_str(self):
        s = ''
        for card in self.pocket:
            s += card.__str__(0) + ', '
        return s
    
    def front_str(self):
        s = ''
        for card in self.pocket:
            s += str(card) + ', '
        return s
    
    def __str__(self):
        pocket = """
           id : {}
           points: {}
           pocket: {}
           
        """.format(names[self.index], self.points, self.pocket_str())
        return pocket
    
    def skulled(self):
        self.clear()
        try:
            self.pocket.remove(self.pocket[randrange(len(self.pocket))])
            return 1
        except:
            return 0
        
    def has_hidden_cards(self):
        return -1 in [card for card in self.public_front()]
    
    def has_cards(self):
        return self.nbr_flowers() + self.has_skull() > 0
    
    def win(self, reward_func):
        self.reward(1, reward_func)
        
    def lose(self, reward_func):
        self.reward(-1, reward_func)
                    
    def play(self, state, legal_moves, forward_func,reward_func):
        if self.reward_pending: 
            self.reward(0, reward_func)
        if 'Human' not in self.name:
                state = format_state(state)
        if legal_moves.count(1) > 1:
            legal_moves = np.array(legal_moves)
            probas = forward_func(state, self.name)
            probas *= legal_moves
            #if not math.isnan(probas):
            try:
                index = list(probas).index(probas.max()) 
            except:
                print(probas)
                print(legal_moves)
                index = 0
        else:
            forward_func(state, self.name, 0)
            index = legal_moves.index(1)
        actions.sort(key = lambda c: c.index)
        try: action = actions[index]
        except:
            print(actions, index, state, legal_moves)
            raise IndexError
        self.reward_pending = action
        if action.type == 'BET' and action.value > self.max_bet: self.max_bet = action.value
        elif action.type == 'CARD' and action.value == 0: self.last_skull = 0
        return action

    def refresh_stats(self):
        self.max_bet = 0
        self.last_skull += 1

    def reward(self, amount, reward_func, action = None):
        try:
            if action == None: action = self.reward_pending.index
        except:
            print('ERROR')
            print(self)
        reward_func(amount, action, self.name)
        self.reward_pending = False
    def pocket_length(self):
        return len(self.pocket)
        
class Env:
    
    def __init__(self, players):
        self.nbr_players = len(players)
        self.players = [Player(player) for player in players]
        self.alive_players = [player for player in self.players]
        self.iters = 1000
        self.next_player = None
        self.order = []

    def __str__(self, g = True):
        players = ''
        for player in self.players:
            players += str(player)
        return players
        
    def relative_fronts(self,player):
        fronts  = [self.order[x].public_front() for x in player.relative_order]
        return fronts
    
    def relative_points(self,player):
        return [self.order[x].points for x in player.relative_order]
    
    def relative_in_game_players(self,player, in_game_players):
        order = [self.order[x] for x in player.relative_order]
        return [x in in_game_players for x in order]

    def relative_hidden_cards(self, player):
        return (1 * np.array([self.order[x].has_hidden_cards() for x in player.relative_order])).tolist()

    def relative_bet_holder(self, player, bet_holder):
        if bet_holder == None: return -1
        if bet_holder in self.order[self.order.index(player):]:
            return self.order.index(bet_holder) - self.order.index(player)
        else:
            return 5 - self.order.index(player) + self.order.index(bet_holder)
        
        
    def relative_pockets(self, player):
        return [self.order[x].pocket_length() for x in player.relative_order]
    
    def relative_bets(self,player):
        return [self.order[x].max_bet for x in player.relative_order]
    
    def relative_last_skull(self,player):
        return [self.order[x].last_skull for x in player.relative_order]

    def state(self,player, bet, bet_holder,in_game_players):
        return {'FLOWERS':player.nbr_flowers(), 'SKULL':player.has_skull(), 'FRONTS':self.relative_fronts(player),
                'OWN_FRONT':player.private_front(), 'HAS_HIDDEN':self.relative_hidden_cards(player),
                'ALIVE':self.relative_in_game_players(player,in_game_players),
                'NBR_CARDS':self.nbr_cards(), 'BET':bet, 'POCKETS':self.relative_pockets(player),
                'BET_HOLDER':self.relative_bet_holder(player, bet_holder), 'POINTS':self.relative_points(player),'LAST SKULLS': self.relative_last_skull(player), 'BETS':self.relative_bets(player)}

    def order_players(self, start = 0, previous_player = 0):
        if start:
            self.order = []
            order = list(range(len(self.players)))
            shuffle(order)
            for x in order:
                new_player = self.players[x]
                self.order.append(new_player)
            for new_player in self.order:
                l = list(range(self.nbr_players))
                l_2 =l[self.order.index(new_player):] + l[:self.order.index(new_player)]
                l_2.remove(self.order.index(new_player))
                new_player.relative_order = l_2
            previous_player = -1
        next_player = previous_player + 1
        if next_player == self.nbr_players: next_player = 0
        order = self.order[next_player:] + self.order[:next_player]
        return order
    
    def reset_players(self):
        for player in self.players:
            player.reset()

    def nbr_cards(self):
        array = np.array([front for front in [player.public_front() for player in self.players]]).flatten()
        return -array.sum()
    
    def clear_players(self):
        for player in self.players: player.clear()

    def kill(self,player):
        player.clear()
        self.alive_players.remove(player)

    def get_choice(self, player, bet, bet_holder, in_game_players):
        state = self.state(player, bet, bet_holder, in_game_players)
        legal_moves = self.get_legal_moves(player, state, in_game_players)
        choice = player.play(state, legal_moves, self.forward, self.reward)
        return choice
    
    def card_stage(self):
        bet = 0
        bet_holder = None
        in_game_players = [player for player in self.alive_players]
        
        while bet == 0 :
            for player in self.order_players():
                
                choice = self.get_choice(player, bet, bet_holder, in_game_players)

                if player in in_game_players:

                    if choice.type == 'CARD' and player.has_cards():
                        if choice.value == 1 and player.nbr_flowers() > 0:
                            player.play_flower()
                        elif choice.value == 0 and player.has_skull():
                            player.play_skull()
                        else:
                            bet = 1
                            bet_holder = player

                    elif choice.value > 0:
                        bet = choice.value
                        bet_holder = player
                        break

                    else:
                        bet = 1
                        bet_holder = player
                        break
                    
        return bet, bet_holder, in_game_players
    
    def bet_stage(self, bet, bet_holder, in_game_players):
        
        nbr_cards = self.nbr_cards()
        restart_index = self.order.index(bet_holder)
        
        while len(in_game_players) > 1:

            for player in self.order_players(previous_player = restart_index):
    
                choice = self.get_choice(player, bet, bet_holder, in_game_players)
                if player in in_game_players:
                    if choice.type == 'BET' and nbr_cards >= choice.value > bet:
                        bet = choice.value
                        bet_holder = player
                        if choice.value == nbr_cards: in_game_players = [player]
                        break

                    else:
                        in_game_players.remove(player)

                    if len(in_game_players) <=1: break

        return bet, bet_holder, in_game_players

    def show_self_stage(self,bet,bet_holder,in_game_players):
        shown_cards = 0
        player = bet_holder
        while bet_holder.has_hidden_cards() and shown_cards <bet:
            card = bet_holder.show()
            if card != 1:
                player.skulled()
                if not player.has_cards():
                    self.kill(player)
                in_game_players = []
                break
            else:
                shown_cards += 1
        return shown_cards, in_game_players
    
    def show_stage(self, bet, bet_holder, in_game_players,shown_cards):
        restart_index = self.order.index(bet_holder)
        while shown_cards < bet and len(in_game_players)>0:

            if shown_cards >= bet: break
            if bet_holder not in in_game_players: break
            for player in self.order_players(previous_player = restart_index):
                choice = self.get_choice(player, bet, bet_holder, in_game_players)
                if player == bet_holder:
                    if choice.type == 'SHOW':
                        card = self.order[player.relative_order[choice.value-1]].show()
                        if card != 1:
                            player.skulled()
                            if not player.has_cards():
                                self.kill(player)
                            in_game_players = []
                            break
                        else:
                            shown_cards += 1       
                    else:
                        player.skulled()
                        if not player.has_cards():
                            self.kill(player)
                        in_game_players = []
                        break
        return shown_cards
    
    def step(self, iters, forward, reward):
        self.forward = forward
        self.reward = reward
        results = []
        for iter in range(iters):
            self.reset_players()
            rnd = 1
            self.alive_players = [player for player in self.players]
            winner = 0
            order = self.order_players(start=1)
            while winner == 0:

                bet, bet_holder, in_game_players = self.card_stage()

                bet, bet_holder, in_game_players = self.bet_stage(bet, bet_holder, in_game_players)
                
                shown_cards, in_game_players = self.show_self_stage(bet, bet_holder, in_game_players)
                    
                shown_cards = self.show_stage(bet, bet_holder, in_game_players,shown_cards)
                
                if shown_cards >= bet:
                    bet_holder.points += 1
                    if bet_holder.points == 2:
                        winner = bet_holder
                if len(self.alive_players) <= 1:
                    break
                self.clear_players()   
            if winner:
                winner.win(self.reward)
                for player in self.players:
                    if player != winner:
                        player.lose(self.reward)
                result = [0,0,0,0]
                result[self.players.index(winner)] = 1
                results.append(result)
            else:
                for player in self.players:
                    player.lose(self.reward)
                results.append([0,0,0,0])
        return results

    def get_legal_moves(self,player, state, in_game_players):
        if player in in_game_players:
            indexes = list(range(22))
            if state['BET_HOLDER'] == 0:
                indexes = indexes[19:]
                x = 0
                for v in state['HAS_HIDDEN']:
                    if not v:
                        indexes.remove(indexes[x])
                        x-=1
                    x+=1
            else:
                indexes = indexes[:state['NBR_CARDS']+2]
                if state['BET'] != 0 or not True in [ x >0 for x in [state['FLOWERS'], state['SKULL']]]:
                    
                    for index in indexes[:state['BET']+3]: indexes.remove(index)
                    indexes.append(18)
                    
                else :
                    indexes = indexes[:19]
                    x = 0
                    while x < len(state['HAS_HIDDEN']):
                        if not state['HAS_HIDDEN'][x] and state['ALIVE'][x]:
                            indexes = indexes[:2]
                            break
                        x+=1
                    if 0 in state['OWN_FRONT']:
                        indexes = indexes[:2]
                    if state['FLOWERS'] < 1:
                        indexes.remove(indexes[1])
                    elif state['SKULL'] == 0:
                        indexes.remove(indexes[0])
        else:
            indexes = [18]
        response = [0 for x in range(22)]
        for i in indexes : response[i] = 1
        return response
    
                
class Env_Process(Process):
    def __init__(self, conn, models):
        Process.__init__(self, target =  Env_Process.step, args =(conn,
                                                                 models))
        self.start()
    def step(conn, models):
        model_names = list(models.keys())
        env = Env(model_names)
        parent_conn, child_conn = Pipe()
        def forward(state,agent_name, forward_pass = 1):
            models[agent_name].send([1,[state, forward_pass], child_conn])
            if forward_pass:
                probas = parent_conn.recv()
                return probas

        def reward(amount, action, agent_name):
            models[agent_name].send([2, [amount, action]])
        results = env.step(25, forward, reward)
        conn.send([3,[results, model_names]]) 
     
            
                    
                
    
            
