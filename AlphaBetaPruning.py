class AlphaBetaPruning:
    def __init__(self, depth, game_state, player):
        self.depth=depth
        self.game_state=game_state
        self.player=player
        self.node_count=0

    def is_terminal(self, state):
         for i in range(3):
            if state[i][0] == state[i][1] == state[i][2] and state[i][0] != ' ':
                return True
            if state[0][i] == state[1][i] == state[2][i] and state[0][i] != ' ':
                return True
       
        if state[0][0] == state[1][1] == state[2][2] and state[0][0] != ' ':
            return True
        if state[0][2] == state[1][1] == state[2][0] and state[0][2] != ' ':
            return True
     
        for row in state:
            if ' ' in row:
                return False
        
        return True
        
    def utility(self, state):
        for i in range(3):
            if state[i][0] == state[i][1] == state[i][2] and state[i][0] != ' ':
                return 1 if state[i][0] == 'X' else -1
            if state[0][i] == state[1][i] == state[2][i] and state[0][i] != ' ':
                return 1 if state[0][i] == 'X' else -1

        if state[0][0] == state[1][1] == state[2][2] and state[0][0] != ' ':
            return 1 if state[0][0] == 'X' else -1
        if state[0][2] == state[1][1] == state[2][0] and state[0][2] != ' ':
            return 1 if state[0][2] == 'X' else -1
    
        return 0
        
    def alphabeta(self, state, depth, alpha, beta, maximizing_player):
        def alphabeta(self, state, depth, alpha, beta, maximizing_player):
        self.node_count += 1  
        if depth == 0 or self.is_terminal(state):
            return self.utility(state)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in self.get_available_moves(state):
                new_state = self.apply_move(state, move, 'X')
                eval = self.alphabeta(new_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.get_available_moves(state):
                new_state = self.apply_move(state, move, 'O')
                eval = self.alphabeta(new_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break 
            return min_eval

    def best_move(self, state):
        best_val = float('-inf')
        best_move = None
        for move in self.get_available_moves(state):
            new_state = self.apply_move(state, move, 'X')
            move_val = self.alphabeta(new_state, self.depth - 1, float('-inf'), float('inf'), False)
            if move_val > best_val:
                best_val = move_val
                best_move = move
        return best_move

