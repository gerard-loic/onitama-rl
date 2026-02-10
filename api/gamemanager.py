from sessionmemory import SessionMemory
import sys
sys.path.append('../onitama/')
from game import Game
from players import HeuristicPlayer, ApiPlayer
from constants import *

players = {
    'heuristic_regular' : HeuristicPlayer(heuristic_function='heuristic_regular')
}

class GameManager:
    @staticmethod
    def create(player:str):
        #Initialisation de P1 (Joueur)
        p1 = ApiPlayer()

        #Initialisation de P2 (Machine)
        if player in players:
            p2 = players[player]
        else:
            raise Exception(f"Player {player} not found !")   

        game = Game(player_one=p1, player_two=p2, verbose=False)
        turn_num = 1
        #Si c'est au tour de p1
        if game.current_player == p2:
            print("On joue le tour de p2")
            game.playGame(return_winner=True, max_turns=200, play_once_only=True)
            turn_num += 1
            
        uid = SessionMemory.createSession(data={'game' : game, 'turn_num' : turn_num})
        gm = GameManager(uid=uid)

        return gm


    def __init__(self, uid:str):
        session = SessionMemory.getSession(sessionId=uid)
        self.uid = uid
        self.game = session['game']
        self.turn_num = session['turn_num']




    def get_game_representation(self):
        ended, winner = self.game.board.game_has_ended()

        return {
            'game_uid' : self.uid,
            'turn_num' : self.turn_num,
            'player_cards' : self.game.board.current_player_cards,
            'opponent_cards' : self.game.board.next_player_cards,
            'neutral_card' :self.game.board.neutral_card,
            'board' : self.game.board.board,
            'ended' : ended,
            'winner' : winner
        }
