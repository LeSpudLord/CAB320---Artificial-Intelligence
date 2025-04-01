
'''

    Sokoban assignment


The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

You are NOT allowed to change the defined interfaces.
In other words, you must fully adhere to the specifications of the 
functions, their arguments and returned values.
Changing the interfacce of a function will likely result in a fail 
for the test of your code. This is not negotiable! 

You have to make sure that your code works with the files provided 
(search.py and sokoban.py) as your code will be tested 
with the original copies of these files. 

Last modified by 2021-08-17  by f.maire@qut.edu.au
- clarifiy some comments, rename some functions
  (and hopefully didn't introduce any bug!)

'''

# You have to make sure that your code works with 
# the files provided (search.py and sokoban.py) as your code will be tested 
# with these files
import search 
import sokoban
from collections import deque


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():

    return [ (11582774, 'Orrin', 'Hatch'), (11734400, 'Michael', 'Pettigrew') ]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def taboo_cells(warehouse):
    '''  
    Identify the taboo cells of a warehouse. A "taboo cell" is by definition
    a cell inside a warehouse such that whenever a box get pushed on such 
    a cell then the puzzle becomes unsolvable. 
    
    Cells outside the warehouse are not taboo. It is a fail to tag one as taboo.
    
    When determining the taboo cells, you must ignore all the existing boxes, 
    only consider the walls and the target  cells.  
    Use only the following rules to determine the taboo cells;
     Rule 1: if a cell is a corner and not a target, then it is a taboo cell.
     Rule 2: all the cells between two corners along a wall are taboo if none of 
             these cells is a target.
    
    @param warehouse: 
        a Warehouse object with a worker inside the warehouse

    @return
       A string representing the warehouse with only the wall cells marked with 
       a '#' and the taboo cells marked with a 'X'.  
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.  
    '''
    #### We need to let the program sort through rows and columns, so set up the warehouse ####

    # Grid workspace dimensions
    ncols = warehouse.ncols
    nrows = warehouse.nrows

    # Set up the grid workspace (list of lists filled with space, ie no walls or objects)
    grid = [[' ' for _ in range(ncols)] for _ in range(nrows)]

    # Initialise walls and targets for quick lookup
    walls = set(warehouse.walls)
    targets = set(warehouse.targets)

    # Mark coordinates of wall pieces
    for (x, y) in walls:
        grid[y][x] = '#' # I am pretty sure its '#' but we will check

    #### NEW: Add flood fill to restrict taboo marking to reachable floor space ####

    def flood_fill_reachable(walls, start, nrows, ncols):
        visited = set()
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            if (x, y) in visited or (x, y) in walls:
                continue
            visited.add((x, y))
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < ncols and 0 <= ny < nrows and (nx, ny) not in visited:
                    queue.append((nx, ny))
        return visited

    reachable = flood_fill_reachable(walls, warehouse.worker, nrows, ncols)

    # Now comes whats considered as "TABOO CELLS (SPACES)" deterministic logic, I am going to do my best to figure out the logic :D

    # Rule 1: Mark corners as taboo cells if they are not targets.
    # A cell is a corner if it is free (i.e. not a wall, not a target)
    # and has walls in two adjacent perpendicular directions.

    for y in range(nrows): # Textbook said imbedded for loops allow the program to cross check? making it dynamically deterministic
        for x in range(ncols):
            if (x, y) not in reachable:
                continue
            if (x, y) in walls or (x, y) in targets:
                continue
            # needs to check all four corner configurations
            top_left = (x - 1 >= 0 and (x - 1, y) in walls) and (y - 1 >= 0 and (x, y - 1) in walls)
            top_right = (x + 1 < ncols and (x + 1, y) in walls) and (y - 1 >= 0 and (x, y - 1) in walls)
            bottom_left = (x - 1 >= 0 and (x - 1, y) in walls) and (y + 1 < nrows and (x, y + 1) in walls)
            bottom_right = (x + 1 < ncols and (x + 1, y) in walls) and (y + 1 < nrows and (x, y + 1) in walls)
            if top_left or top_right or bottom_left or bottom_right:
                grid[y][x] = 'X'

    # Rule 2: For horizontal segments.
    # For each row, if there are two taboo cells with a gap between them
    # and none of the in-between cells is a target, mark all cells in between as taboo.

    for y in range(nrows):
        taboo_indices = [x for x in range(ncols) if grid[y][x] == 'X' and (x, y) in reachable]
        for i in range(len(taboo_indices) - 1):
            x1 = taboo_indices[i]
            x2 = taboo_indices[i + 1]
            if x2 - x1 > 1:
                if all((x, y) not in targets and (x, y) not in walls and (x, y) in reachable for x in range(x1 + 1, x2)):
                    for x in range(x1 + 1, x2):
                        if grid[y][x] != '#': # do not override walls
                            grid[y][x] = 'X' 

    # Rule 2: For vertical segments.
    for x in range(ncols):
        taboo_indices = [y for y in range(nrows) if grid[y][x] == 'X' and (x, y) in reachable]
        for i in range(len(taboo_indices) - 1):
            y1 = taboo_indices[i]
            y2 = taboo_indices[i + 1]
            if y2 - y1 > 1:
                if all((x, y) not in targets and (x, y) not in walls and (x, y) in reachable for y in range(y1 + 1, y2)):
                    for y in range(y1 + 1, y2):
                        if grid[y][x] != '#':
                            grid[y][x] = 'X'

    result = "\n".join("".join(row) for row in grid)
    return result

# We may need to check through the logic make sure it all makes sense
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class SokobanPuzzle(search.Problem):
    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of 
    the provided module 'search.py'. 
    
    '''
    
    #
    #         "INSERT YOUR CODE HERE"
    #
    #     Revisit the sliding puzzle and the pancake puzzle for inspiration!
    #
    #     Note that you will need to add several functions to 
    #     complete this class. For example, a 'result' method is needed
    #     to satisfy the interface of 'search.Problem'.
    #
    #     You are allowed (and encouraged) to use auxiliary functions and classes

    
    def __init__(self, warehouse):
        raise NotImplementedError()

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        """
        raise NotImplementedError

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_elem_action_seq(warehouse, action_seq):
    '''
    
    Determine if the sequence of actions listed in 'action_seq' is legal or not.
    
    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
        
    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
           
    @return
        The string 'Impossible', if one of the action was not valid.
           For example, if the agent tries to push two boxes at the same time,
                        or push a box into a wall.
        Otherwise, if all actions were successful, return                 
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    '''
    
    ##         "INSERT YOUR CODE HERE"
    
    raise NotImplementedError()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_weighted_sokoban(warehouse):
    '''
    This function analyses the given warehouse.
    It returns the two items. The first item is an action sequence solution. 
    The second item is the total cost of this action sequence.
    
    @param 
     warehouse: a valid Warehouse object

    @return
    
        If puzzle cannot be solved 
            return 'Impossible', None
        
        If a solution was found, 
            return S, C 
            where S is a list of actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
            C is the total cost of the action sequence C

    '''
    
    raise NotImplementedError()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

