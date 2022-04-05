import util

class DFS(object):
    def depthFirstSearch(self, problem):
        """
        Search the deepest nodes in the search tree first
        [2nd Edition: p 75, 3rd Edition: p 87]

        Your search algorithm needs to return a list of actions that reaches
        the goal.  Make sure to implement a graph search algorithm
        [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

        To get started, you might want to try some of these simple commands to
        understand the search problem that is being passed in:

        print "Start:", problem.getStartState()
        print "Is the start a goal?", problem.isGoalState(problem.getStartState())
        print "Start's successors:", problem.getSuccessors(problem.getStartState())
        """
        "*** TTU CS3568 YOUR CODE HERE ***"
        stack = util.Stack()
        stack.push((problem.getStartState(),[]))
        visit = set()
        
        while not stack.isEmpty():
            temp = stack.pop()
            if problem.isGoalState(temp[0]):
                return temp[1]
            visit.add(temp[0])

            for pol in problem.getSuccessors(temp[0]):
                if pol[0] not in visit and pol[0] not in stack.list:
                    stack.push((pol[0], temp[1] + [pol[1]]))

        return []
    

class BFS(object):
    def breadthFirstSearch(self, problem):
        "*** TTU CS3568 YOUR CODE HERE ***"
        
        queue = util.Queue()
        visit = {}
        solution = []
        parent = {}
        start = problem.getStartState()
        queue.push((start, 'Undefined', 0))
        visit[start] = 'Undefined'

        # return start state equals goal
        if problem.isGoalState(start):
            return solution

        goal = False;
        while(queue.isEmpty() != True and goal != True):
            temp = queue.pop()
            visit[temp[0]] = temp[1]
            if (problem.isGoalState(temp[0])):
                node = temp[0]
                goal = True
                break
            # for expand node
            for child in problem.getSuccessors(temp[0]):
                if child[0] not in visit.keys() and child[0] not in parent.keys():
                    parent[child[0]] = temp[0]
                    queue.push(child)

        # finding and storing the path
        while(node in parent.keys()):
            previous = parent[node]
            solution.insert(0, visit[node])
            node = previous

        return solution 
            
class UCS(object):
    def uniformCostSearch(self, problem):
        "*** TTU CS3568 YOUR CODE HERE ***"
        queue = util.PriorityQueue()
        visit = {}
        solution = []
        parent = {}
        cost = {}

        start = problem.getStartState()
        queue.push((start, 'Undefined', 0), 0)
        visit[start] = 'Undefined'
        cost[start] = 0

        if problem.isGoalState(start):
            return solution

        goal = False;
        while(queue.isEmpty() != True and goal != True):
            node = queue.pop()
            visit[node[0]] = node[1]
            if problem.isGoalState(node[0]):
                node = node[0]
                goal = True
                break
            for x in problem.getSuccessors(node[0]):
                if x[0] not in visit.keys():
                    priority = node[2] + x[2]
                    if x[0] in cost.keys():
                        if cost[x[0]] <= priority:
                            continue
                    queue.push((x[0], x[1], priority), priority)
                    cost[x[0]] = priority
                    # store successor and its parent
                    parent[x[0]] = node[0]

        # finding and storing the path
        while(node in parent.keys()):
            previous = parent[node]
            solution.insert(0, visit[node])
            node = previous

        return solution
      
    # def uniformCostSearch(self, problem):
    #     "*** TTU CS3568 YOUR CODE HERE ***"
    #     path = []
    #     temp = (problem.getStartState(), 0)
    #     node = temp[0]
    #     cost = temp[1]
        
    #     visit = set()
    #     data = util.PriorityQueue()
    #     data.push((node, path, 0), 0)
        
    #     while not data.isEmpty():
    #         temp = data.pop()
    #         node, path, cost = temp[0], temp[1], temp[2] 
    #         if problem.isGoalState(node):
    #             return path

    #         visit.add(node)
    #         for n in problem.getSuccessors(node):
    #             childNode, childPath, childCost = n[0], n[1], n[2]

    #             if childNode not in visit and childNode not in data.heap:
    #                 data.push((childNode, path + [childPath], cost + childCost), cost + childCost)     
    #     util.raiseNotDefined()
        
class aSearch (object):
    def nullHeuristic( state, problem=None):
        """
        A heuristic function estimates the cost from the current state to the nearest goal in the provided SearchProblem.  This heuristic is trivial.
        """
        return 0
    def aStarSearch(self,problem, heuristic=nullHeuristic):
        "Search the node that has the lowest combined cost and heuristic first."
        "*** TTU CS3568 YOUR CODE HERE ***"
        queue = util.PriorityQueue()        
        visit = {}
        solution = []
        parent = {}
        cost = {}
        
        start = problem.getStartState()
        queue.push((start, 'Undefined', 0), 0)
        visit[start] = 'Undefined'
        cost[start] = 0

        if problem.isGoalState(start):
            return solution

        goal = False;
        while(queue.isEmpty() != True and goal != True):
            node = queue.pop()
            visit[node[0]] = node[1]
            if problem.isGoalState(node[0]):
                node = node[0]
                goal = True
                break
            for x in problem.getSuccessors(node[0]):
                if x[0] not in visit.keys():
                    priority = node[2] + x[2] + heuristic(x[0], problem)
                    if x[0] in cost.keys():
                        if cost[x[0]] <= priority:
                            continue
                    queue.push((x[0], x[1], node[2] + x[2]), priority)
                    cost[x[0]] = priority
                    parent[x[0]] = node[0]

        while(node in parent.keys()):
            previous = parent[node]
            solution.insert(0, visit[node])
            node = previous

        return solution
       