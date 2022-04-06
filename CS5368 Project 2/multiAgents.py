# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        """  the reflex agent compares the manhattan distance between the foodchunk and the position and also ghostpostion to initial position. takes the sum of the absolute difference value 
        newpos[0] & posGhost[0] and newpos[1] & posghost[1] to check if its greater than 1.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
      
        score=successorGameState.getScore()
        foodChunk=newFood.asList()

        for fo in foodChunk:
            distFood = util.manhattanDistance(fo,newPos)
            if (distFood) != 0:
                score=score+(1.0/distFood)


        for g in newGhostStates:
            posGhost=g.getPosition()
            distGhost = util.manhattanDistance(posGhost,newPos)
            if (abs(newPos[1]-posGhost[1]) + abs(newPos[0]-posGhost[0]))>1:
                score=score+(1.0/distGhost)
	    
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
       

        def minValue(state, agentIndex, depth):
            
            legalActions = state.getLegalActions(agentIndex)
            agentNoCount = gameState.getNumAgents()

            if not legalActions:
                return self.evaluationFunction(state)
            if agentIndex == agentNoCount - 1:
                minimumValue =  min(maxValue(state.generateSuccessor(agentIndex, x), agentIndex,  depth) for x in legalActions)
            else:
                minimumValue = min(minValue(state.generateSuccessor(agentIndex, x), agentIndex + 1, depth) for x in legalActions)

            return minimumValue

        def maxValue(state, agentIndex, depth):
             agentIndex = 0
             legalActions = state.getLegalActions(agentIndex)
             if not legalActions  or depth == self.depth:
                 return self.evaluationFunction(state)

             maximumValue =  max(minValue(state.generateSuccessor(agentIndex, x), agentIndex + 1, depth + 1) for x in legalActions)

             return maximumValue

        actionState = gameState.getLegalActions(0)
        Actions = {}
        for y in actionState:
            Actions[y] = minValue(gameState.generateSuccessor(0, y), 1, 1)

        return max(Actions, key=Actions.get)
                
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        
        def minValue(state, agentIndex, depth, alpha, beta):
            
            legalActions = state.getLegalActions(agentIndex)
            agentNo = gameState.getNumAgents()

       
            if not legalActions:
                return self.evaluationFunction(state)

          
            minimumValue = 99999
            intialBeta = beta
           
            if agentIndex == agentNo - 1:
                for x in legalActions:
                    minimumValue =  min(minimumValue, maxValue(state.generateSuccessor(agentIndex, x), \
                    agentIndex,  depth, alpha, intialBeta))
                    if minimumValue < alpha:
                        return minimumValue
                    intialBeta = min(intialBeta, minimumValue)

            else:
                for x in legalActions:
                    minimumValue =  min(minimumValue,minValue(state.generateSuccessor(agentIndex, x),agentIndex + 1, depth, alpha, intialBeta))
                    if minimumValue < alpha:
                        return minimumValue
                    intialBeta = min(intialBeta, minimumValue)

            return minimumValue

        
        def maxValue(state, agentIndex, depth, alpha, beta):
            
            agentIndex = 0
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions  or depth == self.depth:
                return self.evaluationFunction(state)

            initialAlpha = alpha
            maxValue = -99999

            for y in legalActions:
                maxValue = max(maxValue, minValue(state.generateSuccessor(agentIndex, y), agentIndex + 1, depth + 1, initialAlpha, beta) )
                if maxValue > beta:
                    return maxValue
                initialAlpha = max(initialAlpha, maxValue)
            return maxValue

       
        actions = gameState.getLegalActions(0)
        alpha = -99999
        beta = 99999
    
        everyActions = {}
        for x in actions:
            value = minValue(gameState.generateSuccessor(0, x), 1, 1, alpha, beta)
            everyActions[x] = value
            if value > beta:
                return x
            alpha = max(value, alpha)

        return max(everyActions, key=everyActions.get)

        util.raiseNotDefined()



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        

        
        def expectValue(state, agentIndex, depth):
            
            agentNoCount = gameState.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)

            
            if not legalActions:
                return self.evaluationFunction(state)

            
            value = 0
            probabilty = 1.0 / len(legalActions) 
            
            for y in legalActions:
                if agentIndex == agentNoCount - 1:
                    currentExpValue =  maxValue(state.generateSuccessor(agentIndex, y), agentIndex,  depth)
                else:
                    currentExpValue = expectValue(state.generateSuccessor(agentIndex, y), agentIndex + 1, depth)
                value += currentExpValue * probabilty

            return value


        
        def maxValue(state, agentIndex, depth):
            
            agentIndex = 0
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions  or depth == self.depth:
                return self.evaluationFunction(state)

            maxmumValue =  max(expectValue(state.generateSuccessor(agentIndex, y), agentIndex + 1, depth + 1) for y in legalActions)

            return maxmumValue

     
        actions = gameState.getLegalActions(0)
        
        everyActions = {}
        for y in actions:
            everyActions[y] = expectValue(gameState.generateSuccessor(0, y), 1, 1)

        
        return max(everyActions, key=everyActions.get)

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    """  firstly we stored a list for minFoodList with minimum manhattandistance between position & food.same goes for ghost & food distance.
        Assign value foodleftM, capsleftM & foodDisM with random value. Now checks whether game is lost or not with value assigned to respective work.
        lastly we calculate the function  1.0/(foodLeft + 1) * foodLeftM + distGhost + 1.0/(minFoodList + 1) * foodDistM + 1.0/(capsLeft + 1) * capsLeftM + factors to get the necessary output.
    
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()

    minFoodList = float('inf')
    for f in newFood:
        minFoodList = min(minFoodList, manhattanDistance(newPos, f))

    distGhost = 0
    for ghost in currentGameState.getGhostPositions():
        distGhost = manhattanDistance(newPos, ghost)
        if (distGhost < 2):
            return -float('inf')

    foodLeft = currentGameState.getNumFood()
    capsLeft = len(currentGameState.getCapsules())
    foodLeftM = 950050
    capsLeftM = 10000
    foodDistM = 950

    factors = 0
    if currentGameState.isLose():
        factors -= 50000
    elif currentGameState.isWin():
        factors += 50000

    return 1.0/(foodLeft + 1) * foodLeftM + distGhost + 1.0/(minFoodList + 1) * foodDistM + 1.0/(capsLeft + 1) * capsLeftM + factors

    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
