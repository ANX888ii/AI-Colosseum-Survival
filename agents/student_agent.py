# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
from copy import copy
import time
import random
from platform import node

@register_agent("student_agent")
class StudentAgent(Agent):

    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        #0 is the opp direction of 2 since u is the opp direction of d, same for 1 and 3, 1 is the opp direction of 3
        self.opp = {0: 2, 1: 3, 2: 0, 3: 1}
        # dir:         l,      r,       u,      d
        self.dir = ((-1, 0), (0, 1), (1, 0), (0, -1))
        #points
        self.PosPoints = []
        # self start first
        self.turn = True
        self.middle = 0

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
            Implement the step function of your agent here.
            You can use the following variables to access the chess board:
            - chess_board: a numpy array of shape (x_max, y_max, 4)
            - my_pos: a tuple of (x, y)
            - adv_pos: a tuple of (x, y)
            - max_step: an integer

            You should return a tuple of ((x, y), dir),
            where (x, y) is the next position of your agent and dir is the direction of the wall
            you want to put on.

            Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # Some simple code to help you with timing. Consider checking
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        time_taken = time.time() - start_time

        print("My AI's turn took ", time_taken, "seconds.")

        if self.turn:
            self.init_points(chess_board)
            middle_x=(chess_board.shape[0]-1)/2
            middle_y=(chess_board.shape[0]-1)/2

            self.turn = False
            self.middle = (middle_x, middle_y)

        # Get an array of all possible legall moves
        moves = self.actions(chess_board, my_pos, max_step, adv_pos, {})
        final_moves = self.total_a(moves, chess_board)
        #remove all the suicidal moves from the list of array
        final_moves = self.prevent_suicide(my_pos, adv_pos, chess_board, final_moves)
        final_moves = sorted(final_moves, key=lambda x: tuple(np.absolute(tuple(np.subtract(self.middle, x[0])))))

        #exceptions
        try:
            bestMove = 0
            for depth in range(10):
                l = {}
                score = None
                for move in final_moves:
                    chess_board, new_pos = self.take_act(chess_board, move)
                    temp = my_pos
                    score = self.alpha_beta(chess_board, max_step, depth, -np.inf, np.inf, new_pos, adv_pos, False)

                    chess_board = self.remove_act(chess_board, move)
                    my_pos = temp

                    if score == None:
                        break
                    l[move] = score

                # illegal moves: run a random walk
                if score == None:
                    if bestMove == 0:
                        if len(final_moves) == 0:
                            print("executing random walk----------------------------------")
                            return self.random_act(my_pos, adv_pos, max_step, chess_board)
                        else:

                            print("choosing random move from final_moves--------------------------------")
                            return random.choice(final_moves)

                    else:
                        return bestMove

                else:
                    bestMove = max(l, key=l.get)
                    return bestMove

        except:
            if len(final_moves) == 0:
                print("executing random walk 2----------------------------------")
                return self.random_act(my_pos, adv_pos, max_step, chess_board)

            else:
                print("choosing random move from final_moves 2--------------------------------")
                return random.choice(final_moves)

     # Squares near borders are worth less points than squares closer to the center
    def init_points(self, chess_board):
            column = 0
            row = 0
            scale = 2
            points = 0
            size = chess_board.shape[0]

            while row < size:
                a_row = []
                for i in range(size):

                    if (i < size // 2 and size % 2 == 1) or (i <= size / 2 - 1 and size % 2 == 0):

                        if (row >= size // 2 and size % 2 == 1) or (row > size / 2 - 1 and size % 2 == 0):
                            points = round(min(size - 1 - row, i) * scale, 2)
                        else:
                            points = round(min(row, i) * scale, 2)

                    elif (i >= size // 2 and size % 2 == 1) or (i > size / 2 - 1 and size % 2 == 0):
                        if (row >= size // 2 and size % 2 == 1) or (row > size / 2 - 1 and size % 2 == 0):
                            points = round(min(size - 1 - row, size - i - 1) * scale, 2)
                        else:
                            points = round(min(row, size - i - 1) * scale, 2)

                    a_row.append(points)

                self.PosPoints.append(a_row)
                row = row + 1

    def actions(self, chess_board, my_pos, max_step, adv_pos, moves):

        row, col = my_pos
        adv_row, adv_col = adv_pos

        moves[my_pos] = None

        # Check the right
        if (max_step != 0 and not chess_board[row, col, self.dir_map["r"]] and not (adv_row == row and adv_col == col+1)):
            moves.update(self.actions(chess_board, (row, col + 1), max_step-1, adv_pos, moves))

        # Check the left
        if (max_step != 0 and not chess_board[row, col, self.dir_map["l"]] and not (adv_row == row and adv_col == col-1)):
            moves.update(self.actions(chess_board, (row, col - 1), max_step-1, adv_pos, moves))

        # Check the down
        if (max_step != 0 and not chess_board[row, col, self.dir_map["d"]] and not (adv_row == row+1 and adv_col == col)):
            moves.update(self.actions(chess_board, (row + 1, col), max_step-1, adv_pos, moves))

        # Check the up
        if (max_step != 0 and not chess_board[row, col, self.dir_map["u"]] and not (adv_row == row - 1 and adv_col == col)):
            moves.update(self.actions(chess_board, (row - 1, col), max_step - 1, adv_pos, moves))


        return moves

    # --------------------------------------------------------------------------------------------------------------------------------------
    # get array of all possible moves(positions on board where it can move to)
    def total_a(self, moves, chess_board):
        final_moves = []
        list(moves.keys())

        for pos in moves:
            row, col = pos
            possible_dir = self.check_wall(row, col, chess_board)
            for direction in possible_dir:
                new_tuple = ((row, col), self.dir_map[direction])
                final_moves.append(new_tuple)
        return final_moves

    # Returns the walls that are possible for a single square
    def check_wall(self, row, col, chess_board):
        possible_dir = []
        for my_dir in self.dir_map:
            if not chess_board[row, col, self.dir_map[my_dir]]:
                possible_dir.append(my_dir)
        return possible_dir

    def prevent_suicide(self, my_pos, adv_pos, chess_board, moves):
        good_moves = deepcopy(moves)
        for move in moves:
            pos, dir = move
            row, col = pos
            self.barriers(chess_board, row, col, dir, True)

            is_endgame, my_score, adv_score = self.check_ending(
                chess_board, pos, adv_pos)
            if is_endgame and my_score < adv_score:
                good_moves.remove(move)

            self.barriers(chess_board,row, col, dir, False)

        return good_moves

        # returns a tuple (chess_board, new_pos)

    def take_act(self, chess_board, move):
        # copyboard = deepcopy(chess_board)
        new_pos, dir = move
        row, col = new_pos
        self.barriers(chess_board, row, col, dir, True)
        return chess_board, new_pos

        # return chess_board with move undone


    def alpha_beta(self, chess_board, max_step, depth, alpha, beta, my_pos, adv_pos, maxPlayer):
        if time.time() - start_time > move_time:
            # print(time.time()-start_time)
            return None

        end = self.check_ending(chess_board, my_pos, adv_pos)

        if end[0]:
            if end[1] > end[2]:
                return np.inf
            elif end[1] < end[2]:
                return -np.inf
            else:
                return 0

        if depth == 0:
            return self.heuristic_fn(chess_board, my_pos, adv_pos, max_step)

        if maxPlayer:
            maxEval = -np.inf

            nextMoves = self.next_acts(
                chess_board, my_pos, max_step, adv_pos)

            for m in nextMoves:
                temp = my_pos
                chess_board, new_pos = self.take_act(chess_board, m)
                my_pos = new_pos

                score = self.alpha_beta(
                    chess_board, max_step, depth - 1, alpha, beta, my_pos, adv_pos, False)

                chess_board = self.remove_act(chess_board, m)
                my_pos = temp

                if score == None:
                    return None

                maxEval = max(maxEval, score)
                alpha = max(alpha, maxEval)

                if beta <= alpha:
                    break

            return maxEval

        else:   # MinPlayer
            minEval = np.inf
            nextMoves = self.next_acts(
                chess_board, adv_pos, max_step, my_pos)

            for m in nextMoves:
                temp = adv_pos
                chess_board, new_pos = self.take_act(chess_board, m)
                adv_pos = new_pos

                score = self.alpha_beta(
                    chess_board, max_step, depth - 1, alpha, beta, my_pos, adv_pos, True)

                chess_board = self.remove_act(chess_board, m)
                adv_pos = temp

                if score == None:
                    return None

                minEval = min(minEval, score)
                beta = min(beta, minEval)

                if beta <= alpha:
                    break

            return minEval


    def remove_act(self, chess_board, move):
        pos, dir = move
        row, col = pos
        self.barriers(chess_board, row, col, dir, False)
        return chess_board

    def random_act(self, my_pos, adv_pos, max_step, chess_board):
        init_pos = deepcopy(my_pos)
        steps = np.random.randint(0, max_step + 1)
        # Put Barrier
        dir = np.random.randint(0, 4)
        row, col = my_pos

        # Random Walk
        for a in range(steps):
            row, col = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = ((-1, 0), (0, 1), (1, 0), (0, -1))[dir]
            my_pos = (row + m_r, col + m_c)

            # Special Case enclosed by Adversary
            m = 0
            while chess_board[row, col , dir] or my_pos == adv_pos:
                m += 1
                if m > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = ((-1, 0), (0, 1), (1, 0), (0, -1))[dir]
                my_pos = (row + m_r, col + m_c)

            if m > 300:
                my_pos = init_pos
                break

        while chess_board[row, col, dir]:
            dir = np.random.randint(0, 4)
        return my_pos, dir

    # --------------------------------------------------------------------------------------------------------------------------------------

    #verify if there is an end in the game and update the score of each agent
    def check_ending(self, chess_board, my_pos, adv_pos):
        # Union-Find
        parent = {}
        for r in range(chess_board.shape[0]):
            for c in range(chess_board.shape[0]):
                parent[(r, c)] = (r, c)

        def find(pos):
            if parent[pos] != pos:
                parent[pos] = find(parent[pos])
            return parent[pos]

        def union(pos1, pos2):
            parent[pos1] = pos2

        for r in range(chess_board.shape[0]):
            for c in range(chess_board.shape[0]):
                for dir, move in enumerate(
                    ((-1, 0), (0, 1), (1, 0), (0, -1))[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir+1]:
                        continue

                    pos_a = find((r, c))
                    move0=move[0]
                    move1=move[1]
                    pos_b = find((r + move0, c + move1))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(chess_board.shape[0]):
            for c in range(chess_board.shape[0]):
                find((r, c))

        p0_r = find(my_pos)
        p1_r = find(adv_pos)

        my_score = list(parent.values()).count(p0_r)
        adv_score = list(parent.values()).count(p1_r)

        if p0_r == p1_r:
            return False, my_score, adv_score
        player_win = None
        win_blocks = -1
        if my_score > adv_score:
            player_win = 0
            win_blocks = my_score
        elif my_score < adv_score:
            player_win = 1
            win_blocks = adv_score
        else:
            player_win = -1  # Tie

        return True, my_score, adv_score

    # --------------------------------------------------------------------------------------------------------------------------------------

    # Returns a number of walls around the player n squares away (ONLY RETURNS THE NUMBER OF OUTSIDE WALLS, NOT INSIDE ONES)

    def check_outside_walls(self, my_pos, chess_board, n):
        count = 0
        startRow, startColumn = my_pos
        maxCoord = chess_board.shape[0]

        # Check top-right border + top square
        row = startRow - n
        column = startColumn
        myCheck = 0
        while column < maxCoord and row >= 0 and column <= startColumn + n:
            if chess_board[row, column, 0]:
                count = count + 1

            column = column + 1
            myCheck = myCheck + 1

        # Need this to check if one side got cut off early (check ipad for further explanation)
        if myCheck - 1 != n and startRow - n >= 0:
            count = count + 1

        # Check top-left border excluding top square
        row = startRow - n
        column = startColumn - 1
        myCheck = 1
        while column >= 0 and row >= 0 and column >= startColumn - n:
            if chess_board[row, column, 0]:
                count = count + 1

            column = column - 1
            myCheck = myCheck + 1

        # Need this to verify if one side got cut off very soon
        if myCheck - 1 != n and startRow - n >= 0:
            count = count + 1

        # verify the right square and the right-up border
        row = startRow
        column = startColumn + n
        myCheck = 0
        while column < maxCoord and row >= 0 and row >= startRow - n:
            if chess_board[row, column, 1]:
                count = count + 1

            row = row - 1
            myCheck = myCheck + 1

        # to verify if one side got cut off very soon
        if myCheck - 1 != n and startColumn + n < maxCoord:
            count = count + 1

        # Check right-down border excluding right square
        row = startRow + 1
        column = startColumn + n
        myCheck = 1
        while column < maxCoord and row < maxCoord and row <= startRow + n:
            if chess_board[row, column, 1]:
                count = count + 1

            row = row + 1
            myCheck = myCheck + 1

        # to verify if one side got cut off very soon
        if myCheck - 1 != n and startColumn + n < maxCoord:
            count = count + 1

        # verify bottom-right border + bottom square
        row = startRow + n
        column = startColumn
        myCheck = 0
        while column < maxCoord and row < maxCoord and column <= startColumn + n:
            if chess_board[row, column, 2]:
                count = count + 1

            column = column + 1
            myCheck = myCheck + 1

        # Need this to verify if one side got cut off very soon
        if myCheck - 1 != n and startRow + n < maxCoord:
            count = count + 1

        # verify bottom-left border excluding bottom square
        row = startRow + n
        column = startColumn - 1
        myCheck = 1
        while column >= 0 and row < maxCoord and column >= startColumn - n:
            if chess_board[row, column, 2]:
                count = count + 1

            column += 1
            myCheck += 1

        # Need this to verify if one side got cut off very soon
        if myCheck - 1 != n and startRow + n < maxCoord:
            count = count + 1

        # verify left-up border + left square
        row = startRow
        column = startColumn - n
        myCheck = 0
        while column >= 0 and row >= 0 and row >= startRow - n:
            if chess_board[row, column, 3]:
                count += 1

            row -= 1
            myCheck += 1

        # Need this to verify if one side got cut off very soon
        if myCheck - 1 != n and startColumn - n >= 0:
            count += 1

        # verify left-down border excluding left square
        row = startRow + 1
        column = startColumn - n
        myCheck = 1
        while column >= 0 and row < maxCoord and row <= startRow + n:
            if chess_board[row, column, 3]:
                count += 1

            row += 1
            myCheck += 1

        # Need this to verify if one side got cut off very soon
        if myCheck - 1 != n and startColumn - n >= 0:
            count += 1

        return count
    # Returns a number of walls around the player n squares away (ONLY RETURNS THE NUMBER OF INSIDE WALLS, NOT OUTSIDE ONES)

    def check_inside_walls(self, my_pos, chess_board, n):
        count = 0
        startRow, startColumn = my_pos
        maxCoord = chess_board.shape[0]

        # Check top-right border
        row = startRow - n
        column = startColumn + 1
        while column < maxCoord and row >= 0 and column <= startColumn + n:
            # left border
            if chess_board[row, column, 3]:
                count += 1

            column += 1

        #top-left border
        row = startRow - n
        column = startColumn - 1

        while column >= 0 and row >= 0 and column >= startColumn - n:
            # right border
            if chess_board[row, column, 1]:
                count += 1

            column -= 1

        # right-up border
        row = startRow - 1
        column = startColumn + n

        while column < maxCoord and row >= 0 and row >= startRow - n:
            # bottom border
            if chess_board[row, column, 2]:
                count+= 1

            row -= 1

        #  right-down border
        row = startRow + 1
        column = startColumn + n

        while column < maxCoord and row < maxCoord and row <= startRow + n:
            #  top border
            if chess_board[row, column, 0]:
                count += 1

            row += 1

        # bottom-right border
        row = startRow + n
        column = startColumn + 1
        myCheck = 0
        while column < maxCoord and row < maxCoord and column <= startColumn + n:
            # left border
            if chess_board[row, column, 3]:
                count += 1

            column += 1

        # bottom-left border
        row = startRow + n
        column = startColumn - 1

        while column >= 0 and row < maxCoord and column >= startColumn - n:
            # right border
            if chess_board[row, column, 1]:
                count += 1

            column-= 1

        # left-up border
        row = startRow - 1
        column = startColumn - n
        myCheck = 0
        while column >= 0 and row >= 0 and row >= startRow - n:
            # bottom border
            if chess_board[row, column, 2]:
                count += 1

            row -= 1

        # verify left-down border
        row = startRow + 1
        column = startColumn - n

        while column >= 0 and row < maxCoord and row <= startRow + n:
            # verify the top border
            if chess_board[row, column, 0]:
                count += 1

            row += 1

        return count
    # returns all surrounding walls around the player up to n squares away

    #verify surrounding walls
    def check_surroundings_walls(self, my_pos, chess_board, n):
        count = 0

        for i in range(n+1):
            count = count + \
                self.check_surrounding_outside_walls(
                    my_pos, chess_board, i)/(2 * i+1)
            if i > 0:
                count = count + \
                    self.check_inside_walls(
                        my_pos, chess_board, i)/(2 * i+1)

        return count

# -----------------------------------------------------------------------------------------------------------------
    def barriers(self, chess_board, row, col, dir, exist):

        # Set the barrier to True
        chess_board[row, col, dir] = exist
        # Set the opposite barrier to True
        move = self.dir[dir]
        chess_board[row + move[0], col + move[1], self.opp[dir]] = exist


# --------------------------------------------------------------------------------------------------------------------------------------

    # verify if there is a temporary win
    def check_temp_win(self, my_pos, adv_pos, chess_board, moves):
        mate = []
        for move in moves:
            pos, dir = move
            row, col = pos

            # Set the barrier to True
            self.barriers(chess_board, row, col, dir, True)

            is_endgame, my_score, adv_score = self.check_ending(chess_board, pos, adv_pos)

            if my_score < adv_score:
                self.barriers(chess_board, row, col, dir, False)
            else:
                mate.append(move)
                self.barriers(chess_board, row, col, dir, False)
                return mate
        return []


    # If the opponent is surrounded by 2 walls, return the square the move that places a wall s.t. the opponent must flee towards a border
    # Returns 0 if the opponent is not surrounded by exactly 2 walls

    # glowling gears:
    def box_opponent(self, adv_pos, chess_board):
        if self.check_surroundings_walls(adv_pos, chess_board, 0) != 2:
            return 0

        row, col = adv_pos
        possible_dir = self.check_wall(row, col, chess_board)

        maxCoord = chess_board.shape[0]
        minDist = (100, 0)  # default: (distance, direction)

        for dir in possible_dir:
            row, col = adv_pos
            count = 0  # Counts number of squares from adv_pos to a border of the chess board
            # Get the opposite direction
            opp_dir = self.opp[self.dir_map[dir]]

            # While the coordinates are within the bounds of the board...
            while maxCoord > row and row >= 0 and maxCoord > col and col >= 0:
                col += self.dir[opp_dir][1]
                row += self.dir[opp_dir][0]
                count += 1

            if minDist[0] > count:
                minDist = (count, self.dir_map[dir])

        # Now we have the wall that we want to place. We must get the move for our agent
        row, col = adv_pos
        dir = minDist[1]

        # Move one square towards dir
        col += self.dir[dir][1]
        row += self.dir[dir][0]

        # Reverse the direction
        dir = self.opp[dir]

        # Return the move
        return ((row, col), dir)

    # I am between 2 walls, remove moves that will push myself towards a border
    def remove_box_myself(self, my_pos, chess_board):
        if self.check_surroundings_walls(my_pos, chess_board, 0) != 2:
            return 0

        row, col = my_pos
        possible_dir = self.check_wall(row, col, chess_board)

        minDist = (100, 0)  # default: (distance, direction)
        maxCoord = chess_board.shape[0]

        for dir in possible_dir:
            row, col = my_pos
            count = 0   # Counts number of squares from my_pos to a border of the chess board
            # Get the opposite direction
            opp_dir = self.opp[self.dir_map[dir]]

            # While the coordinates are within the bounds of the board...
            while row < maxCoord and row >= 0 and col < maxCoord and col >= 0:
                row += self.dir[opp_dir][0]
                col += self.dir[opp_dir][1]
                count += 1

            if count > minDist[0]:
                minDist = (count, self.dir_map[dir])

        # Return the move
        return ((row, col), minDist[1])



# -----------------------------------------------------------------------------------------------------------------

    #heuristic function
    def heuristic_fn(self, chess_board, my_pos, adv_pos, max_step):

        move_score = self.heuristic_score_act(chess_board, my_pos, adv_pos, max_step)*10

        score_walls = self.h_walls_score(my_pos, adv_pos, chess_board)*2

        r_adv, c_adv = adv_pos
        score_position_adv = self.PosPoints[r_adv][c_adv]

        row, col = my_pos
        score_position_me = self.PosPoints[row][col]

        score_position = score_position_me - score_position_adv

        return round(move_score + score_walls + score_position, 3)

    #heuristic score
    def heuristic_score_act(self, chess_board, my_pos, adv_pos, max_step):
        my_moves = self.actions(chess_board, my_pos, max_step, adv_pos, {})
        nb1 = len(self.total_a(my_moves, chess_board))
        adv_moves = self.actions(chess_board, adv_pos, max_step, adv_pos, {})
        nb2 = len(self.total_a(adv_moves, chess_board))

        return (nb1 - nb2)

    # check_surroundings_walls(self, my_pos, chess_board, n)
    def h_walls_score(self, my_pos, adv_pos, chess_board):
        chess_board_shape0=chess_board.shape[0]
        n = chess_board_shape0 // 4

        # total number of points = 4*n
        my_result = 0
        adv_result = 0

        nbOfWalls = self.check_surroundings_walls(
            adv_pos, chess_board, n)


        if nbOfWalls > 4 * n:
            my_result = 5
        elif nbOfWalls > 3 * n:
            my_result = 4
        elif nbOfWalls > 2 * n:
            my_result = 2.5
        elif nbOfWalls > n:
            my_result = 1
        else:
            my_result = 0

        adv_nbOfWalls = self.check_surroundings_walls(
            my_pos, chess_board, n)

        if adv_nbOfWalls > 4 * n:
            adv_result = 5
        elif adv_nbOfWalls > 3 * n:
            adv_result = 4
        elif adv_nbOfWalls > 2 * n:
            adv_result = 2.5
        elif adv_nbOfWalls > n:
            adv_result = 1
        else:
            adv_result = 0

        return adv_result - my_result

    # -----------------------------------------------------------------------------------------------------------------
    # initialize a graph of depth n
    def next_acts(self, chess_board, my_pos, max_step, adv_pos):
        moves = self.actions(chess_board, my_pos, max_step, adv_pos, {})
        return self.total_a(moves, chess_board)








