
from players import Player, HumanPlayer, RandomPlayer, HeuristicPlayer, LookAheadHeuristicPlayer
from constants import *
from board import Board
from game import Game, GameSession
from pathlib import Path
import pickle
import numpy as np

class DataTrainer:
    def __init__(self):
        pass

    def save_experience(self, player:Player, state:list):
        pass

    def close(self, winner:Player):
        pass

class RegularDataTrainer(DataTrainer):

    @staticmethod
    def getTrainedData(filepath:str):
        all_data = []
        with open(filepath, 'rb') as f:
            while True:
                try:
                    batch = pickle.load(f)
                    all_data.extend(batch)
                except EOFError:
                    break
        return all_data
    

    def __init__(self, p1:Player, p2:Player, p1_record:bool, p2_record:bool, save_only_wins:bool, x_file_destination:str, y_file_destination:str, override:bool=False):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.p1_record = p1_record
        self.p2_record = p2_record
        self.save_only_wins = save_only_wins
        self.x_file_destination = x_file_destination
        self.y_file_destination = y_file_destination

        #Initialisation du cache
        self._init_cache()

        #Si suppression
        if override:
            fichier_x = Path(self.x_file_destination)
            fichier_y = Path(self.y_file_destination)
            if fichier_x.exists():
                fichier_x.unlink()
                print(f"File {self.x_file_destination} deleted !")
            if fichier_y.exists():
                fichier_y.unlink()
                print(f"File {self.y_file_destination} deleted !")

    def _init_cache(self):
        self.cache_states_p1 = []
        self.cache_actions_p1 = []
        self.cache_states_p2 = []
        self.cache_actions_p2 = []

    def save_experience(self, player:Player, state:list, action:list):
        super().save_experience(player, state)

        t = np.array(state)

        if (player == self.p1 and self.p1_record):
            self.cache_states_p1.append(state)
            self.cache_actions_p1.append(action)
        elif (player == self.p2 and self.p2_record):
            self.cache_states_p2.append(state)
            self.cache_actions_p2.append(action)

    def close(self, winner:Player):
        super().close(winner)
    
        x_to_write = []
        y_to_write = []

        if self.save_only_wins:
            if winner == self.p1:
                x_to_write += self.cache_states_p1
                y_to_write += self.cache_actions_p1
            elif winner == self.p2:
                x_to_write += self.cache_states_p2
                y_to_write += self.cache_actions_p2
        else:
            x_to_write += self.cache_states_p1
            x_to_write += self.cache_states_p2
            y_to_write += self.cache_actions_p1
            y_to_write += self.cache_actions_p2

        with open(self.x_file_destination, 'ab') as f:
            pickle.dump(x_to_write, f)
        with open(self.y_file_destination, 'ab') as f:
            pickle.dump(y_to_write, f)
        
        self._init_cache()




if __name__ == "__main__":
    
    training_plan = [
        (
            LookAheadHeuristicPlayer(max_depth=2, heuristic_function="heuristic_aggressive"),
            LookAheadHeuristicPlayer(max_depth=2, heuristic_function="heuristic_defensive"),
            "agressive2-vs-defensive2"
        ),
        (
            LookAheadHeuristicPlayer(max_depth=2),
            LookAheadHeuristicPlayer(max_depth=2, heuristic_function="heuristic_defensive"),
            "regular2-vs-defensive2"
        ),
        (
            LookAheadHeuristicPlayer(max_depth=2, heuristic_function="heuristic_aggressive"),
            LookAheadHeuristicPlayer(max_depth=2, heuristic_function="heuristic_mobility"),
            "agressive2-vs-mobility2"
        ),
        (
            LookAheadHeuristicPlayer(max_depth=2, heuristic_function="heuristic_positional"),
            LookAheadHeuristicPlayer(max_depth=2, heuristic_function="heuristic_mobility"),
            "positional2-vs-mobility2"
        ),
        (
            LookAheadHeuristicPlayer(max_depth=2, heuristic_function="heuristic_noisy"),
            LookAheadHeuristicPlayer(max_depth=2, heuristic_function="heuristic_mobility"),
            "noisy2-vs-mobility2"
        ),
        (
            LookAheadHeuristicPlayer(max_depth=2, heuristic_function="heuristic_noisy"),
            LookAheadHeuristicPlayer(max_depth=2, heuristic_function="heuristic_positional"),
            "noisy2-vs-positional2"
        ),
        (
            LookAheadHeuristicPlayer(max_depth=3, heuristic_function="heuristic_defensive"),
            LookAheadHeuristicPlayer(max_depth=3, heuristic_function="heuristic_positional"),
            "defensive2-vs-positional2"
        ),
    ]

    i = 1
    for p1, p2, filename in training_plan:
        print(f"Training session {i}")
        trainer = RegularDataTrainer(
            p1=p1,
            p2=p2,
            p1_record=True,
            p2_record=True,
            save_only_wins=True,
            x_file_destination=f"../data/{filename}-states.pkl",
            y_file_destination=f"../data/{filename}-actions.pkl",
            override=True
        )
        gameSession = GameSession(player_one=p1, player_two=p2, number_of_games=5000, trainer=trainer)
        gameSession.start()
        print(gameSession.getStats())

    
    


    """
    

    all = RegularDataTrainer.getTrainedData(filepath="../data/training-data-heuristic-vs-laheuristic3-actions.pkl")
    print(len(all))


    all2 = RegularDataTrainer.getTrainedData(filepath="../data/training-data-heuristic-vs-laheuristic2-actions.pkl")
    print(len(all2))
    print(all2[1])
    """