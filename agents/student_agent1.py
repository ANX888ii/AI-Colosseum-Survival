# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import random
import math


@register_agent("student_agent1")
class StudentAgent1(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent1, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.values = (0, 1, 2, 3)
        self.INF = np.inf
        self.NINF = -np.inf
        self.max_iteration = 1

    def getAllowedMoves(self, my_pos, adv_pos, max_step, board):
        allowed_moves = []
        x, y = my_pos

        # Base case: step size = 0
        if max_step == 0:
            for d in range(4):
                if (not board[x, y, d]) and (not adv_pos == my_pos):
                    allowed_moves.append((x, y, d))
            return allowed_moves
        # Step case: step size > 0
        else:
            # Get all allowed wall placement on current position
            allowed_moves += self.getAllowedMoves(my_pos, adv_pos, 0, board)

            # Move 1 step all allowed direction (at most 4)
            for d in range(4):
                if ((not board[x, y, d]) and (not adv_pos == my_pos)):
                    new_pos = (x + self.moves[d][0], y + self.moves[d][1])
                    allowed_moves += self.getAllowedMoves(new_pos, adv_pos, max_step - 1, board)
            # Return a list without duplicates
            return sorted(list(dict.fromkeys(allowed_moves)))

    def getScores(self, my_pos, adv_pos, board):
        board_size = len(board)

        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        return not p0_score == p1_score, p0_score, p1_score

    def getMax(self, my_pos, adv_pos, board, alpha, beta, max_step, max_iteration, alpha_move=None):
        if max_iteration == 0:
            return beta, alpha_move

        end, my_score, adv_score = self.getScores(my_pos, adv_pos, board)
        # For base case, handle the end game scenario
        # If game ends, return defined utility as follows
        utility = my_score - adv_score
        if end:
            return utility, alpha_move
        
        # Step case
        # Get allowed moves for this agent
        allowed_moves = self.getAllowedMoves(my_pos, adv_pos, max_step, board)
        for move in allowed_moves:
            # Return if winning move found
            tmp_board = deepcopy(board)
            tmp_board[move[0], move[1], move[2]] = True
            end, my_score, adv_score = self.getScores((move[0], move[1]), adv_pos, tmp_board)
            utility = my_score - adv_score
            if end and utility > 0:
                return utility, move

            # Update board
            tmp_board = deepcopy(board)
            tmp_board[move[0], move[1], move[2]] = True
            utility = self.getMin((move[0], move[1]), adv_pos, tmp_board, alpha, beta, max_step, max_iteration, move)[0]
            # If pruning condition reached, immediately returns
            if utility > beta:
                return beta, alpha_move
            # Change if found better move
            elif utility >= alpha:
                alpha = utility
                alpha_move = move
            # Else remain unchanged
            else:
                continue

        return alpha, alpha_move
        
    def getMin(self, my_pos, adv_pos, board, alpha, beta, max_step, max_iteration, beta_move=None):
        end, my_score, adv_score = self.getScores(my_pos, adv_pos, board)
        # For base case, handle the end game scenario
        # If game ends, return defined utility as follows
        utility = my_score - adv_score
        if end:
            return utility, beta_move
        
        # Step case
        # Get allowed moves for opponent
        allowed_moves = self.getAllowedMoves(adv_pos, my_pos, max_step, board)
        for move in allowed_moves :
            # Update board
            tmp_board = deepcopy(board)
            tmp_board[move[0], move[1], move[2]] = True
            utility = self.getMax(my_pos, (move[0], move[1]), tmp_board, alpha, beta, max_step, max_iteration - 1, move)[0]
            # If pruning condition reached, immediately returns
            if utility < alpha:
                return beta, beta_move
            # Change if found better move
            elif utility <= beta:
                beta = utility
                beta_move = move
            # Else remain unchanged
            else:
                continue
        
        return beta, beta_move

    def getDistance(self, pos1, pos2):
        return math.sqrt((pos2[1] - pos1[1]) ** 2 + (pos2[0] - pos1[0]) ** 2)

    def refine(self, allowed_moves):
        count = 0
        prev_pos = -1, -1
        prev_d = -1
        refined_moves = []

        for move in allowed_moves:
            current_pos = move[0], move[1]
            current_d = move[2]

            if prev_pos == current_pos:
                refined_moves.append((prev_pos[0], prev_pos[1], prev_d))
                count += 1
            else:
                # If only 1 wall placement is available in move, then DON'T DO IT! It will trap yourself
                # So only append move where more than 1 wall placement is available
                if count > 1:
                    refined_moves.append((prev_pos[0], prev_pos[1], prev_d))
                count = 0
            
            prev_pos = current_pos
            prev_d = current_d
        
        return refined_moves

    def heuristicMove(self, allowed_moves, adv_pos, board):
        target = adv_pos
        dist = self.INF
        closest_move = random.choice(allowed_moves)

        # Select all positions that are the closest to target
        """
        refined_moves = self.refine(allowed_moves)
        if len(refined_moves) == 0:
            print("Random happened!")
            return random_move # A random choice
        """
        
        # Return winning move
        for move in allowed_moves:
            tmp_board = deepcopy(board)
            tmp_board[move[0], move[1], move[2]] = True
            end, my_score, adv_score = self.getScores((move[0], move[1]), adv_pos, tmp_board)
            utility = my_score - adv_score
            if end and utility > 0:
                return move
            
            pos = move[0], move[1]

            new_dist = self.getDistance(pos, target)
            if new_dist < dist:
                closest_move = move
                dist = new_dist
        
        heuristic_moves = []
        for move in allowed_moves:
            pos = move[0], move[1]
            if pos == (closest_move[0], closest_move[1]):
                heuristic_moves.append(move)
        
        """
        if len(heuristic_moves) <= 2:
            print("Random happened!")
            return random_move
        """
        
        # Decide which direction is best, notice that with refine() function, len(heuristic_moves) > 1 !
        for move in heuristic_moves:
            ratio = (adv_pos[0] - move[0], adv_pos[1] - move[1])
            if ratio[0] < 0 and (ratio[0] <= ratio[1] and ratio[1] < abs(ratio[0])):
                value = 0
            elif ratio[1] > 0 and (-ratio[1] <= ratio[0] and ratio[0] < ratio[1]):
                value = 1
            elif ratio[0] > 0 and (ratio[0] >= ratio[1] and ratio[1] > -ratio[0]):
                value = 2
            elif ratio[1] < 0 and (abs(ratio[1]) >= ratio[0] and ratio[0] > ratio[1]):
                value = 3
            else:
                # Corner case but this never will happen
                value = 0
        
        diff = self.INF
        for move in heuristic_moves:
            new_diff = abs(value - move[2])

            if new_diff < diff:
                heuristic_move = move
                diff = new_diff

        return heuristic_move

    def checkSelfTrapping(self, board, move, adv_pos):
        tmp_board = deepcopy(board)
        tmp_board[move[0], move[1], move[2]] = True
        score = self.getScores((move[0], move[1]), adv_pos, tmp_board)
        allowed_moves = self.getAllowedMoves((move[0], move[1]), adv_pos, 0, tmp_board)

        if (len(allowed_moves) == 0) or (score[1] < score[2]):
            return True
        else:
            return False

    def step(self, board, my_pos, adv_pos, max_step):
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

        # Allowed moves
        allowed_moves = self.getAllowedMoves(my_pos, adv_pos, max_step, board)
        random_move = random.choice(allowed_moves)
        #print(allowed_moves)

        score = self.getScores(my_pos, adv_pos, board)
        max_stuff = self.getMax(my_pos, adv_pos, board, self.NINF, self.INF, max_step, self.max_iteration, None)
        min_stuff = self.getMin(my_pos, adv_pos, board, self.NINF, self.INF, max_step, self.max_iteration, None)
        #print(score)
        print("Utility and best move according to alpha-beta: ", max_stuff)
        #print(min_stuff)

        """
        # In case that alpha beta is unsuccessful in finding the best move with limited depth
        # Choose the next move according to the following heuristic:
        # next_step <- the move in allowed_moves that is closest to the center
        # next_dir <- the direction facing opponent
        """
        if (max_stuff[0] == self.INF):
            #print("Going heuristic!")
            move = self.heuristicMove(allowed_moves, adv_pos, board)
        else:
            move = max_stuff[1]

        print("Move chosen (by alpha-beta or by heuristic): ", move)
        #stop = input()

        # Final check, if got bad move somehow, do random
        while self.checkSelfTrapping(board, move, adv_pos) and len(allowed_moves) > 0:
            print("Self trap!")
            #stop = input()
            allowed_moves.remove(move)
            if len(allowed_moves) == 0:
                move = random_move
            else:
                move = random.choice(allowed_moves)
            print("Another move to avoid self trapping: ", move)
        
        if allowed_moves == None:
            move = random_move

        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # Return (next_x, next_y), next_direction
        next_step = (move[0], move[1])
        next_dir = move[2]
        return next_step, next_dir
