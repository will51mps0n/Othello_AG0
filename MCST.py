import logging

import coloredlogs
import math
import queue
import torch
import random
import os
import sys
import numpy as np

from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn
from utils import *
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

from tqdm import tqdm

from Arena import Arena

from IPython.display import clear_output
import time

import logging
import math
import queue

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Vs = {}

        self.Qsa = {}
        self.Nsa = {}
        self.Ps = {}
        self.Ns = {}

        # this is the only member variable you'll have to use. It'll be used in select()
        self.visited = set() # all "state" positions we have seen so far

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        self.MCTsearch(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def gameEnded(self, canonicalBoard):
      """
      This function determines if the current board position is the end of the game.

      Returns:
          gameReward: a value that returns 0 if the game hasn't ended, 1 if the player won, -1 if the player lost
      """

      gameReward = self.game.getGameEnded(canonicalBoard, 1)
      return gameReward

    def predict(self, state, canonicalBoard):
        """
        A wrapper to perform predictions and necessary policy masking for the code to work.
        The key idea is to call this function to return an initial policy vector and value from the neural network
        instead of needing a rollout

        Returns:
            r: the reward given by the neural network
        """
        self.Ps[state], val = self.nnet.predict(canonicalBoard)
        valids = self.game.getValidMoves(canonicalBoard, 1)
        self.Ps[state] = self.Ps[state] * valids
        sum_Ps_s = np.sum(self.Ps[state])
        if sum_Ps_s > 0:
            self.Ps[state] /= sum_Ps_s
        else:
            log.error("All valid moves were masked, doing a workaround.")
            self.Ps[state] = self.Ps[state] + valids
            self.Ps[state] /= np.sum(self.Ps[state])

        self.Vs[state] = valids
        self.Ns[state] = 0
        return val

    def getValidActions(self, state):
        """
        Generates the valid actions from the avialable actions. Actions are given as a list of integers.
        The integers represent which spot in the board to place an Othello disc.
        To see a (x, y) representation of an action, you can do "x, y = (int(action/self.game.n), action%self.game.n)"

        Returns:
            validActions: all valid actions you can take in terms of a list of integers
        """

        validActions = []
        for action in range(self.game.getActionSize()):
            if self.Vs[state][action]:
                validActions.append(action)
        return validActions

    def nextState(self, canonicalBoard, action):
        """
        Gets the next board state given the action

        Returns:
            nextBoard: the next board state given the action
        """

        nextState, nextPlayer = self.game.getNextState(canonicalBoard, 1, action)
        nextState = self.game.getCanonicalForm(nextState, nextPlayer)
        return nextState

    def getConfidenceVal(self, state, action):
        if (state, action) not in self.Qsa:
            self.Qsa[(state, action)] = 0
            self.Nsa[(state, action)] = 0

        u = self.Qsa[(state, action)] + self.args.cpuct * self.Ps[state][action] * math.sqrt(self.Ns[state]) / (
                    1 + self.Nsa[(state, action)])

        return u

    def updateValues(self, r, state, action):
        self.Qsa[(state, action)] = (self.Nsa[(state, action)] * self.Qsa[(state, action)] + r) / (self.Nsa[(state, action)] + 1)
        self.Nsa[(state, action)] += 1
        self.Ns[state] += 1

    def expand(self, state):
        self.visited.add(state)

    #todo
    def select(self, state, board):
        """Serves as the select phase of the MCTS algorithm, should return a tuple of (state, board, action, reward)"""
        r = self.gameEnded(board)

        if r != 0:
          return None, None, None, -r


        if state not in self.visited:
            self.expand(state)
            r2 = self.simulate(state, board)
            return None, None, None, -r2

        u = np.NINF
        bestAction = None

        for actionPrime in self.getValidActions(state):
            newConfVal = self.getConfidenceVal(state, actionPrime)
            if newConfVal > u:
                # i think this was the bug and why it is late...
                bestAction = actionPrime
                u = newConfVal

        board = self.nextState(board, bestAction)
        state = self.game.stringRepresentation(board)
        return state, board, bestAction, 0

    #TODO
    def backpropagate(self, seq):
        """This function uses the seq that you build and maintain in self.MCTsearch
        and iterates through it to propagate values into search tree"""
        r = 0 
        while not seq.empty():
            # This method retrieves front of Lifo.Queue and pops, the structure for this tuple should be defined by you
            curr_state_tuple = seq.get()
            state, action, reward = curr_state_tuple

            if reward != 0:
                r = reward
            else:
                self.updateValues(r, state, action)
                r*=-1
        return

    def simulate(self, state, board):
        reward = self.predict(state, board)
        return reward


    def MCTsearch(self, start_board):
        """
        This function performs MCTS. The action chosen at each node is one that
        has the maximum upper confidence bound.
        """

        start_state = self.game.stringRepresentation(start_board)
        r = self.gameEnded(start_board)

        for _ in range(self.args.numMCTSSims):
            state = start_state
            board = start_board
            sequence = queue.LifoQueue()
            r = 0

            while r == 0:  # game not over
                #next_result = self.select(state, board)
                nextState, nextBoard, action, r = self.select(state, board)

                sequence.put((state, action, r))
                state = nextState
                board = nextBoard
            #end while
            self.backpropagate(sequence)

        return start_board
    
    log = logging.getLogger(__name__)

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False

        # self.actionsTaken is the variable that keeps track of what actions you take in the selection phase.
        # The goal is to make sure this matches the instructor's results
        self.actionsTaken = []

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            # normally the action chosen is random. But we have seeded numpy so it's deterministic
            action = np.random.choice(len(pi), p=pi)

            # IMPORTANT: this line keeps track of what actions you take in the selection phase
            # This is what you'll be graded on in gradescope
            move = (int(action/self.game.n), action%self.game.n)
            self.actionsTaken[-1].append(move)

            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            log.info(f'Starting Iter #{i} ...')
            # start a new list of actions taken for the next iteration
            self.actionsTaken.append([])

            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)
                    # the executeEpisode calls will be made here
                    iterationTrainExamples += self.executeEpisode()

                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            self.saveTrainExamples(i - 1)

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            self.skipFirstSelfPlay = True
seed = 492
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  

args = dotdict({
    'numIters': 1,
    'numEps': 2,              
    'tempThreshold': 15,
    'updateThreshold': 0.6,     
    'maxlenOfQueue': 200000,    
    'numMCTSSims': 5,          
    'arenaCompare': 25,        
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

log.info('Loading %s...', Game.__name__)
g = Game(6)

log.info('Loading %s...', nn.__name__)
nnet = nn(g)

if args.load_model:
    log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
    nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
else:
    log.warning('Not loading a checkpoint!')

log.info('Loading the Coach...')
c = Coach(g, nnet, args)

if args.load_model:
    log.info("Loading 'trainExamples' from file...")
    c.loadTrainExamples()

log.info('Loading %s...', Game.__name__)
g = Game(6)

log.info('Loading %s...', nn.__name__)
nnet = nn(g)

if args.load_model:
    log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
    nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
else:
    log.warning('Not loading a checkpoint!')

log.info('Loading the Coach...')
# c = Coach(g, nnet, args)

if args.load_model:
    log.info("Loading 'trainExamples' from file...")
    c.loadTrainExamples()
log.info('Starting the learning process')
c.learn()

instrResults = [[(2, 1), (3, 1), (4, 2), (5, 1), (4, 1), (3, 0), (4, 0), (1, 1), (1, 3), (1, 2), (0, 1), (1, 0), (5, 2), (0, 3), (2, 0), (0, 0), (2, 4), (0, 2), (0, 4), (1, 5), (2, 5), (5, 3), (4, 4), (3, 4), (0, 5), (1, 4), (3, 5), (4, 5), (5, 5), (5, 4), (5, 0), (4, 3), (3, 4), (2, 4), (1, 2), (2, 1), (3, 0), (4, 4), (3, 5), (4, 1), (1, 3), (2, 0), (3, 1), (2, 5), (5, 2), (0, 3), (0, 4), (4, 0), (5, 4), (0, 5), (0, 2), (4, 2), (1, 4), (4, 5), (1, 0), (1, 5), (5, 0), (1, 1), (0, 1), (5, 3), (4, 3), (0, 0), (5, 5), (5, 1)]]
print(c.actionsTaken)

# VISUALIZATION: 
iteration = 0
board = g.getInitBoard()

g.display(board)
for i, move in enumerate(c.actionsTaken[iteration]):
  # converts (x, y) to an action the game understands
  action = move[0]*g.n + move[1]
  board = g.getNextState(board, 1 if i % 2 == 0 else -1, action)[0]

  time.sleep(2)
  clear_output(wait=False)
  g.display(board)

print("game finished!")

'''
This output :
   0 1 2 3 4 5 
-----------------------
0 |X X O X X X |
1 |O O O O O X |
2 |O O O X O X |
3 |O X O X X X |
4 |O X X X X X |
5 |O O O O O O |
-----------------------
'''