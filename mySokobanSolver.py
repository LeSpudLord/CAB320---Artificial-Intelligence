
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
from scipy.optimize import linear_sum_assignment
import heapq

class SokobanState:
    def __init__(self, worker, boxes):
        self.worker = worker
        self.boxes = tuple(sorted(boxes))
    
    def __eq__(self, other):
        return isinstance(other, SokobanState) and self.worker == other.worker and self.boxes == other.boxes
    
    def __hash__(self):
        return hash((self.worker, self.boxes))
    
    def __lt__(self, other):
        return (self.worker, self.boxes) < (other.worker, other.boxes)
    
    def __repr__(self):
        return f"SokobanState(worker={self.worker}, boxes={self.boxes})"

DIRS = {'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1)}

def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(11582774, 'Orrin', 'Hatch'), (11734400, 'Michael', 'Pettigrew')]

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
    ncols, nrows = warehouse.ncols, warehouse.nrows
    walls = set(warehouse.walls)
    targets = set(warehouse.targets)

    # Step 1: compute floor from walls + targets (ignore boxes, player)
    floor = set()
    for y in range(nrows):
        for x in range(ncols):
            if (x, y) not in walls:
                floor.add((x, y))

    # Step 2: flood fill from outside to get invalid floor (open space)
    def get_outside_spaces():
        visited = set()
        queue = deque()
        for x in range(ncols):
            queue.append((x, 0))
            queue.append((x, nrows - 1))
        for y in range(nrows):
            queue.append((0, y))
            queue.append((ncols - 1, y))
        while queue:
            x, y = queue.popleft()
            if (x, y) in visited or (x, y) not in floor:
                continue
            visited.add((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < ncols and 0 <= ny < nrows:
                    queue.append((nx, ny))
        return visited

    outside = get_outside_spaces()
    valid_floor = floor - outside

    # Step 3: build base grid
    grid = [[' ' for _ in range(ncols)] for _ in range(nrows)]
    for (x, y) in walls:
        grid[y][x] = '#'

    taboo = set()

    def is_corner(x, y):
        if (x, y) not in valid_floor or (x, y) in targets:
            return False
        left = (x - 1, y) in walls
        right = (x + 1, y) in walls
        up = (x, y - 1) in walls
        down = (x, y + 1) in walls
        return (left and up) or (right and up) or (left and down) or (right and down)

    # Rule 1: mark corners
    for y in range(1, nrows - 1):
        for x in range(1, ncols - 1):
            if is_corner(x, y):
                taboo.add((x, y))

    for (x, y) in taboo:
        grid[y][x] = 'X'

    # Rule 2: fill between Xs hugging a wall
    # Horizontal
    for y in range(1, nrows - 1):
        x = 1
        while x < ncols - 1:
            if grid[y][x] == 'X':
                start = x
                x += 1
                while x < ncols - 1 and grid[y][x] != 'X':
                    x += 1
                if x < ncols - 1 and grid[y][x] == 'X':
                    wall_above = all((x0, y - 1) in walls for x0 in range(start + 1, x))
                    wall_below = all((x0, y + 1) in walls for x0 in range(start + 1, x))
                    no_targets = all((x0, y) not in targets and grid[y][x0] != '#' for x0 in range(start + 1, x))
                    if no_targets and (wall_above or wall_below):
                        for x0 in range(start + 1, x):
                            grid[y][x0] = 'X'
            else:
                x += 1

    # Vertical
    for x in range(1, ncols - 1):
        y = 1
        while y < nrows - 1:
            if grid[y][x] == 'X':
                start = y
                y += 1
                while y < nrows - 1 and grid[y][x] != 'X':
                    y += 1
                if y < nrows - 1 and grid[y][x] == 'X':
                    wall_left = all((x - 1, y0) in walls for y0 in range(start + 1, y))
                    wall_right = all((x + 1, y0) in walls for y0 in range(start + 1, y))
                    no_targets = all((x, y0) not in targets and grid[y0][x] != '#' for y0 in range(start + 1, y))
                    if no_targets and (wall_left or wall_right):
                        for y0 in range(start + 1, y):
                            grid[y0][x] = 'X'
            else:
                y += 1

    return '\n'.join(''.join(row) for row in grid)





class SokobanPuzzle(search.Problem):
    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of 
    the provided module 'search.py'. 
    
    '''
    def __init__(self, warehouse):
        self.warehouse = warehouse
        self.walls = set(warehouse.walls)
        self.targets = set(warehouse.targets)
        self.initial_boxes = tuple((i, pos) for i, pos in enumerate(warehouse.boxes))
        self.weights_dict = {i: w for i, w in enumerate(warehouse.weights)}
        self.initial = SokobanState(warehouse.worker, self.initial_boxes)
        self.h = search.memoize(self.heuristic, slot='h')
        self.box_id_map = {pos: i for i, pos in enumerate(self.initial_boxes)}
        self.weights = warehouse.weights or [1] * len(warehouse.boxes)
        self._reachable_cache = {}
        self.taboo = self._compute_taboo_cells()
        self.actions_called = 0
        
    
    def heuristic(self, node):
        # Let:
        #   B = set of current box positions
        #   T = set of target positions
        #   w_i = weight of box i
        #   d(p, q) = Manhattan distance between points p and q

        worker, boxes = node.state.worker, node.state.boxes
        targets = list(self.targets)
        cost_matrix = []
        push_penalty = 0.5
        worker_dist_weight = 0.4

        # Build cost matrix C where:
        #   C[i][j] = w_i * d(box_i, target_j)
        for box_id, box in boxes:
            weight = self.weights_dict.get(box_id, 1)
            row = []
            for target in targets:
                dist = abs(box[0] - target[0]) + abs(box[1] - target[1])  # d(box, target)
                row.append(dist * weight)  # weighted cost
            cost_matrix.append(row)

        # Use Hungarian algorithm to find assignment A minimizing total cost:
        #   min Σ C[i][A[i]] over all box-target pairings
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment_cost = sum(cost_matrix[i][j] for i, j in zip(row_ind, col_ind))

        # Estimate push effort (number of boxes not yet on targets)
        #   push_penalty ≈ 0.5 * |{ b ∈ B : b ∉ T }|
        push_cost_estimate = len([b for _, b in boxes if b not in self.targets]) * push_penalty

        # Compute min distance from worker to any box:
        #   d_worker = min d(worker, box_i)
        min_worker_box_dist = min(
            abs(worker[0] - b[0]) + abs(worker[1] - b[1])
            for _, b in boxes
        )


        # Return the total estimated cost:
        return assignment_cost + worker_dist_weight * min_worker_box_dist +  push_cost_estimate

    def _compute_taboo_cells(self):
        # Run taboo_cells() and parse the result into a set of taboo positions
        taboo_string = taboo_cells(self.warehouse)
        taboo_set = set()
        lines = taboo_string.splitlines()
        for y, row in enumerate(lines):
            for x, cell in enumerate(row):
                if cell == 'X':
                    taboo_set.add((x, y))
        return taboo_set

    def reachable_positions(self, worker, boxes):
        # Use a cached result if available
        key = (worker, frozenset(pos for _, pos in boxes))
        if key in self._reachable_cache:
            return self._reachable_cache[key]

        visited = set()
        queue = deque([worker])
        boxes_set = set(pos for _, pos in boxes)

        while queue:
            pos = queue.popleft()
            if pos in visited or pos in self.walls or pos in boxes_set:
                continue
            visited.add(pos)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if new_pos not in visited and new_pos not in self.walls and new_pos not in boxes_set:
                    queue.append(new_pos)

        # Cache the result
        self._reachable_cache[key] = visited
        return visited
    

    
    # Returns all legal push actions for the given state
    def actions(self, state):
        self.actions_called += 1

        worker, boxes = state.worker, state.boxes
        boxes_pos = [pos for _, pos in boxes]
        boxes_set = set(boxes_pos)

        # Get reachable positions (cached)
        reachable = self.reachable_positions(worker, boxes)

        actions = []

        for i, box in boxes:
            x, y = box
            for direction, (dx, dy) in DIRS.items():
                behind = (x - dx, y - dy)
                target = (x + dx, y + dy)

                # Skip push if target is blocked
                if target in self.walls or target in boxes_set:
                    continue
                if is_stuck(target, boxes_pos, self.walls, self.targets):
                    continue
                if target in self.taboo:
                    continue
                # If the worker can reach the "behind" cell
                if behind in reachable:
                    actions.append(((i, box), direction))

        return actions
   
    # Applies a single push action and returns the resulting new state
    def result(self, state, action):
        

        (box_id, box), direction = action
        dx, dy = DIRS[direction]

        # Update the position of the pushed box; all other boxes stay in place
        new_boxes = [
            (bid, (box[0] + dx, box[1] + dy)) if bid == box_id else (bid, bpos)
            for bid, bpos in state.boxes
        ]

        # Worker ends up in the previous box location (where the push happened)
        return SokobanState(box, new_boxes)

    # Returns True if all boxes are on target positions
    def goal_test(self, state):
        return set(pos for _, pos in state.boxes) == self.targets

    # Calculates the cost of performing an action from state1 to state2
    def path_cost(self, c, state1, action, state2):
        (box_id, _), _ = action
        weight = self.weights_dict.get(box_id, 1)  # Get the box's weight
        walk_cost = 1  # Fixed cost for the worker's movement
        return c + walk_cost + weight  # Total cost includes movement and push effort


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
    # Define movement directions with corresponding coordinate deltas
    DIRS = {'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1)}

    # Get initial worker and box positions from the warehouse
    worker = warehouse.worker
    boxes = list(warehouse.boxes)
    walls = set(warehouse.walls)

    # Simulate each action in the sequence
    for action in action_seq:
        # Reject any invalid action
        if action not in DIRS:
            return "Impossible"

        dx, dy = DIRS[action]
        next_pos = (worker[0] + dx, worker[1] + dy)

        # If worker walks into a wall, the sequence is invalid
        if next_pos in walls:
            return "Impossible"

        # If the worker tries to push a box
        if next_pos in boxes:
            box_next = (next_pos[0] + dx, next_pos[1] + dy)

            # If the box would be pushed into a wall or another box, the move is invalid
            if box_next in walls or box_next in boxes:
                return "Impossible"

            # Update the box's position
            boxes.remove(next_pos)
            boxes.append(box_next)

        # Move the worker to the new position (step or push)
        worker = next_pos

    # Recreate the warehouse with the new state and return its string representation
    final = warehouse.copy(worker=worker, boxes=boxes)
    
    return str(final)



def manhattan(p1, p2):
    # Calculate the Manhattan distance between two points
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

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
    # Construct the Sokoban problem instance with weighted boxes
    puzzle = SokobanPuzzle(warehouse)

    # Use A* search to find a solution node using our custom heuristic
    solution_node = my_astar_graph_search(puzzle, h=puzzle.h)

    # If no solution is found, return 'Impossible' and no cost
    if solution_node is None:
        return 'Impossible', None

    # Extract the macro-level plan (box pushes) and its associated cost
    actions = solution_node.solution()
    cost = solution_node.path_cost

    # Convert the macro plan (box pushes) into a full step-by-step action plan
    full_plan = expand_macro_plan(warehouse, actions)

    # If expansion failed, stop early
    if full_plan == "Impossible":
        return "Impossible", None

    # Count the additional steps the worker had to take (e.g. walking to each push location)
    step_cost = (
        full_plan.count('Left') + 
        full_plan.count('Right') + 
        full_plan.count('Up') + 
        full_plan.count('Down') - 
        len(actions)
    )

    # Combine the macro-level push cost with additional movement cost
    total_cost = cost + step_cost
    print("Total actions() calls:", puzzle.actions_called)
    return full_plan, total_cost


def find_path(walls, boxes, start, goal, path_cache=None):
    if path_cache is not None:
        key = (start, goal, frozenset(boxes))
        if key in path_cache:
            return path_cache[key]

    DIRS = {'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1)}
    frontier = [(manhattan(start, goal), 0, start, [])]
    visited = set()

    while frontier:
        _, cost, current, path = heapq.heappop(frontier)
        if current == goal:
            if path_cache is not None:
                path_cache[key] = path
            return path
        if current in visited:
            continue
        visited.add(current)
        for direction, (dx, dy) in DIRS.items():
            next_pos = (current[0] + dx, current[1] + dy)
            if next_pos in walls or next_pos in boxes or next_pos in visited:
                continue
            heapq.heappush(frontier, (
                cost + 1 + manhattan(next_pos, goal),
                cost + 1,
                next_pos,
                path + [direction]
            ))

    if path_cache is not None:
        path_cache[key] = None
    return None


# Converts a macro-level plan (just box pushes) into a full step-by-step action sequence,
# including all movements the worker needs to make before each push.
def expand_macro_plan(warehouse, macro_plan):
    path_cache = {}
    # Direction vectors for movement commands
    DIRS = {'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1)}

    # Start from the initial position of the worker
    worker = warehouse.worker

    # Keep track of box positions using their IDs (as in the initial warehouse)
    boxes = [(i, pos) for i, pos in enumerate(warehouse.boxes)]

    # This will store the full list of primitive actions: movements + pushes
    full_plan = []

    # For each macro-action (a box push), determine the movement needed to set it up
    for ((box_id, box), direction) in macro_plan:
        dx, dy = DIRS[direction]

        # Determine the position the worker must stand in to push this box
        push_from = (box[0] - dx, box[1] - dy)

        # Plan a path for the worker to reach the push position
        path_to_push = find_path(
            set(warehouse.walls),
            set(pos for _, pos in boxes),
            worker,
            push_from,
            path_cache
        )

        # If the worker can't reach the push position, the plan fails
        if path_to_push is None:
            return "Impossible"

        # Add the movement path to the full plan
        full_plan.extend(path_to_push)

        # Add the actual push direction (the macro-action)
        full_plan.append(direction)

        # Update the worker's new position (now where the box was)
        worker = box

        # Update the box's new position after the push
        box_index = boxes.index((box_id, box))
        boxes[box_index] = (box_id, (box[0] + dx, box[1] + dy))

    return full_plan

def is_stuck(pos, boxes, walls, targets):
    x, y = pos
    boxes_set = set(boxes)

    # 1. Classic corner deadlock
    if ((x - 1, y) in walls and (x, y - 1) in walls) or \
       ((x + 1, y) in walls and (x, y - 1) in walls) or \
       ((x - 1, y) in walls and (x, y + 1) in walls) or \
       ((x + 1, y) in walls and (x, y + 1) in walls):
        return pos not in targets

    # 2. Box against wall, short scan for no target in line
    wall_scan_range = 3

    # Vertical wall check (horizontal movement)
    #if (x - 1, y) in walls or (x + 1, y) in walls:
    #    if all((i, y) not in targets for i in range(x - wall_scan_range, x + wall_scan_range + 1)
    #           if (i, y) not in walls):
    #        return True

    # Horizontal wall check (vertical movement)
    #if (x, y - 1) in walls or (x, y + 1) in walls:
    #    if all((x, j) not in targets for j in range(y - wall_scan_range, y + wall_scan_range + 1)
    #           if (x, j) not in walls):
    #        return True/*

    # 3. 2-box freeze check
    adjacent = [
        ((x + 1, y), (x, y + 1)),  # box right and box down
        ((x - 1, y), (x, y + 1)),  # box left and box down
        ((x + 1, y), (x, y - 1)),  # box right and box up
        ((x - 1, y), (x, y - 1)),  # box left and box up
        ((x + 1, y), (x - 1, y)),  # horizontal pair
        ((x, y + 1), (x, y - 1)),  # vertical pair
    ]
    for a, b in adjacent:
        if a in boxes_set and b in boxes_set:
            if ((a[0] == b[0] == x) and ((x - 1, y) in walls or (x + 1, y) in walls)) or \
               ((a[1] == b[1] == y) and ((x, y - 1) in walls or (x, y + 1) in walls)):
                if pos not in targets and a not in targets and b not in targets:
                    return True

    return False



def my_astar_graph_search(problem, h=None):
    """Custom A* implementation with cost-based pruning (can't modify search.py)."""
    h = search.memoize(h or problem.h, slot='h')
    node = search.Node(problem.initial)

    frontier = search.PriorityQueue(order='min', f=lambda n: n.path_cost + h(n))
    frontier.append(node)

    explored = set()
    state_cost = {node.state: 0}

    while frontier:
        node = frontier.pop()

        if problem.goal_test(node.state):
            return node

        if node.state in explored:
            continue
        explored.add(node.state)

        for child in node.expand(problem):
            s = child.state
            new_cost = child.path_cost

            if s not in state_cost or new_cost < state_cost[s]:
                state_cost[s] = new_cost
                frontier.append(child)

    return None

def test_taboo_cells_simple():
    wh = sokoban.Warehouse()
    wh.from_lines([
        "##########",
        "# $      #",
        "# . @    #",
        "#        #",
        "##########"
    ])
    output = taboo_cells(wh)
    print("Taboo Cell Output:")
    print(output)

def test_taboo_cells_corner():
    wh = sokoban.Warehouse()
    wh.from_lines([
        "#######",
        "# $   #",
        "#   @ #",
        "#  .  #",
        "#######"
    ])
    output = taboo_cells(wh)
    print("Taboo Cell Corner Test:")
    print(output)

def test_taboo_cells_from_file(filename):
    """
    Load a warehouse layout from a text file and print the taboo cell map.
    The file should contain the warehouse layout using standard Sokoban symbols."""
    wh = sokoban.Warehouse()
    with open(filename, 'r') as f:
        lines = f.readlines()
        wh.from_lines([line.rstrip('\n') for line in lines])

    output = taboo_cells(wh)
    print(f"Taboo Cells for: {filename}")
    print(output)

print("\n--- Running Taboo Cell Tests ---")
test_taboo_cells_simple()
test_taboo_cells_corner()

test_taboo_cells_from_file("warehouses\warehouse_6n.txt")
