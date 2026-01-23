import numpy as np
from card import Card
from collections import namedtuple
from constants import *
import random

Action = namedtuple('Action', ['from_pos', 'card_idx', 'to_pos', 'move_idx'])

class Board:
    @staticmethod
    def cell_to_string(cell:int):
        return STR_MAPPING[cell]
    
    @staticmethod
    def rotate_180(board:list):
        # Inverse l'ordre des colonnes ET l'ordre dans chaque colonne
        return [col[::-1] for col in board[::-1]]


    #------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, player_to_move:int, current_player_cards:list, next_player_cards:list, neutral_card:Card, board:list=None):
        self.first_player = player_to_move
        self.current_player = player_to_move
        self.next_player = PLAYER_TWO_POSITION if self.current_player == PLAYER_ONE_POSITION else PLAYER_ONE_POSITION
        self.current_player_cards = current_player_cards
        self.next_player_cards = next_player_cards
        self.neutral_card = neutral_card

        self.current_player_student = PLAYER_ONE_STUDENT if self.current_player == PLAYER_ONE_POSITION else PLAYER_TWO_STUDENT
        self.current_player_master = PLAYER_ONE_MASTER if self.current_player == PLAYER_ONE_POSITION else PLAYER_TWO_MASTER
        self.next_player_student = PLAYER_TWO_STUDENT if self.current_player == PLAYER_ONE_POSITION else PLAYER_ONE_STUDENT
        self.next_player_master = PLAYER_TWO_MASTER if self.current_player == PLAYER_ONE_POSITION else PLAYER_ONE_MASTER
        self.current_player_temple = (2, 4)
        self.next_player_temple = (2, 0)

        self.board = [[EMPTY_CELL] * 5 for _ in range(5)] if board is None else board


    """
    Retourne les mouvements possibles, du point de vue de l'utilisateur courant
    """
    def get_available_moves(self):
        moves = []
        #Quelles sont les pieces qui nous interessent ?
        pieces = [self.current_player_student, self.current_player_master]

        #On parcourt le plateau à la recherche des pièces
        for col in range(5):
            for row in range(5):
                #On vérifie que c'est une pièce du joueur
                if self.board[col][row] in pieces:
                    #On créé les actions pour chaque carte
                    for card in self.current_player_cards:
                        cardMoves = card.get_moves_from_position(position=(col, row))
                        for moveCol, moveRow, move_idx in cardMoves:
                            #On vérifie qu'on a pas déjà une pièce sur la case
                            if not self.board[moveCol][moveRow] in pieces:
                                #On ajoute l'actions
                                moves.append(Action((col, row), card.idx, (moveCol, moveRow), move_idx))
        return moves

    
    """
    Joue un coup et retourne sa représentation
    """
    def play_move(self, action:Action):
        last_move = {
            'board_to_resume' : [],
            'neutral_card' : None,
            'current_player_cards' : []
        }

        #Position de départ
        from_x, from_y = action.from_pos

        #Position d'arrivée
        to_x, to_y = action.to_pos

        current_player_pieces = [self.current_player_student, self.current_player_master]
        next_player_pieces = [self.next_player_student, self.next_player_master]
        piece_to_move = self.board[from_x][from_y]

        assert self.board[to_x][to_y] not in current_player_pieces, f"The cell {to_x, to_y} is not valid : current player already there. "
        
        #On garde en mémoire pour pouvoir annuler le coup
        last_move['board_to_resume'].append((from_x, from_y, self.board[from_x][from_y]))
        last_move['board_to_resume'].append((to_x, to_y, self.board[to_x][to_y]))
        last_move['neutral_card'] = self.neutral_card
        for card in self.current_player_cards:
            last_move['current_player_cards'].append(card)

        #On place la pièce sur la nouvelle case
        self.board[to_x][to_y] = piece_to_move
        #On vide la case d'origine
        self.board[from_x][from_y] = EMPTY_CELL

        #On fait tourner la carte utilisée (elle va aller en neutre)
        for i in range(len(self.current_player_cards)):
            if self.current_player_cards[i].idx == action.card_idx:
                break
        self.current_player_cards.append(self.neutral_card)
        self.neutral_card = self.current_player_cards.pop(i)

        #On échange les joueurs
        self.current_player, self.next_player = self.next_player, self.current_player
        self.current_player_student, self.next_player_student = self.next_player_student, self.current_player_student
        self.current_player_master, self.next_player_master = self.next_player_master, self.current_player_master
        self.current_player_cards, self.next_player_cards = self.next_player_cards, self.current_player_cards

        #On retourne le plateau
        self.board = Board.rotate_180(board=self.board)

        return last_move

    def play_default_move(self):
        #On mélange pour aléatoirement choisir une carte qui sera donnée à l'adversaire
        random.shuffle(self.current_player_cards)

        old_neutral = self.neutral_card
        old_current_cards = self.current_player_cards.copy()

        self.current_player_cards.append(self.neutral_card)
        self.neutral_card = self.current_player_cards.pop(0)

        #On échange les joueurs
        self.current_player, self.next_player = self.next_player, self.current_player
        self.current_player_student, self.next_player_student = self.next_player_student, self.current_player_student
        self.current_player_master, self.next_player_master = self.next_player_master, self.current_player_master
        self.current_player_cards, self.next_player_cards = self.next_player_cards, self.current_player_cards

        #On retourne le plateau
        self.board = Board.rotate_180(board=self.board)

        return {'neutral_card': old_neutral, 'current_player_cards': old_current_cards}

    def cancel_default_move(self, last_move:dict=None):
        #On retourne le plateau
        self.board = Board.rotate_180(board=self.board)

        #On échange les joueurs
        self.current_player, self.next_player = self.next_player, self.current_player
        self.current_player_student, self.next_player_student = self.next_player_student, self.current_player_student
        self.current_player_master, self.next_player_master = self.next_player_master, self.current_player_master
        self.current_player_cards, self.next_player_cards = self.next_player_cards, self.current_player_cards

        if last_move:
            self.neutral_card = last_move['neutral_card']
            self.current_player_cards = last_move['current_player_cards']

    
    """
    Annule un coup d'après sa représentation
    """
    def cancel_last_move(self, last_move:dict):
        if last_move is None:
            print("Nothing to cancel !")
            return
        
        #On retourne le plateau
        self.board = Board.rotate_180(board=self.board)
        
        #On réinitialise le board
        for col, row, pawn in last_move['board_to_resume']:
            self.board[col][row] = pawn

        #On échange les joueurs
        self.current_player, self.next_player = self.next_player, self.current_player
        self.current_player_student, self.next_player_student = self.next_player_student, self.current_player_student
        self.current_player_master, self.next_player_master = self.next_player_master, self.current_player_master
        self.current_player_cards, self.next_player_cards = self.next_player_cards, self.current_player_cards

        self.neutral_card = last_move['neutral_card']
        self.current_player_cards = last_move['current_player_cards']


    """
    Retourne si le jeu est terminé, et si il y a un gagnant
    """
    def game_has_ended(self):

        #Si le maitre du joueur courant est sur le temple de l'opposant
        if self.board[self.next_player_temple[0]][self.next_player_temple[1]] == self.current_player_master:
            return True, self.current_player
        
        #Si le maitre de l'opposant est sur le temple du joueur courant
        if self.board[self.current_player_temple[0]][self.current_player_temple[1]] == self.next_player_master:
            return True, self.next_player
        
        #Si le joueur courant n'a plus son maitre
        if not any(self.current_player_master in ligne for ligne in self.board):
            return True, self.next_player
        
        #Si l'opposant n'a plus son maitre
        if not any(self.next_player_master in ligne for ligne in self.board):
            return True, self.current_player

        return False, None
    

    """
    Retourne la position du maitre
    """
    def get_master_position(self, player_position:int):
        code = PLAYER_ONE_MASTER if player_position == PLAYER_ONE_POSITION else PLAYER_TWO_MASTER
        for col in range(5):
            for row in range(5):
                if self.board[col][row] == code:
                    return (col, row)
        return (None, None)


    """
    Retourne une évaluation heuristique de l'état
    """
    def heuristic_evaluation(self, from_current_player_point_of_view:bool=True, verbose:bool=False):

        #On s'adapte au point de vue demandé
        p1 = self.current_player
        p2 = PLAYER_TWO_POSITION if p1 == PLAYER_ONE_POSITION else PLAYER_ONE_POSITION

        p1_student = PLAYER_ONE_STUDENT if p1 == PLAYER_ONE_POSITION else PLAYER_TWO_STUDENT
        p2_student = PLAYER_TWO_STUDENT if p1 == PLAYER_ONE_POSITION else PLAYER_ONE_STUDENT
        p1_master = PLAYER_ONE_MASTER if p1 == PLAYER_ONE_POSITION else PLAYER_TWO_MASTER
        p2_master = PLAYER_TWO_MASTER if p1 == PLAYER_ONE_POSITION else PLAYER_ONE_MASTER
        p1_temple = PLAYER_ONE_TEMPLE if p1 == PLAYER_ONE_POSITION else PLAYER_TWO_TEMPLE
        p2_temple = PLAYER_TWO_TEMPLE if p1 == PLAYER_ONE_POSITION else PLAYER_ONE_TEMPLE

        
        if from_current_player_point_of_view is False:
            p1, p2, p1_student, p2_student, p1_master, p2_master, p1_temple, p2_temple = p2, p1, p2_student, p1_student, p2_master, p1_master, p2_temple, p1_temple


        #Si la partie est perdue
        game_ended, winner = self.game_has_ended()
        if game_ended:
            if winner == p1:
                if verbose:
                    print(f"Score for winning : 1000")
                return 1000
            else:
                if verbose:
                    print(f"Score for lossing : -1000")
                return -1000

        #Différence en nombre d'étudiant
        p1_nb_students = sum(valeur == p1_student for ligne in self.board for valeur in ligne)
        p2_nb_students = sum(valeur == p2_student for ligne in self.board for valeur in ligne)
        score_diff_students = (p1_nb_students-p2_nb_students)*100

        #Distance du maitre de P1 par rapport au temple de P2 (positif)
        master_col, master_row = self.get_master_position(player_position=p1)
        temple_col, temple_row = p2_temple
        distance = abs(master_col - temple_col) + abs(master_row - temple_row)
        score_distance_master_current_player = (8 - distance) * 20

        #Distance du maitre de P2 par rapport au temple de P1 (négtif)
        master_col, master_row = self.get_master_position(player_position=p2)
        temple_col, temple_row = p1_temple
        distance = abs(master_col - temple_col) + abs(master_row - temple_row)
        score_distance_master_next_player = ((8 - distance) * 20)*-1
        
        score = score_diff_students+score_distance_master_current_player+score_distance_master_next_player

        if verbose:
            print(f"Score student difference : {score_diff_students}")
            print(f"Score distance current player master from opponent temple : {score_distance_master_current_player}")
            print(f"Score distance opponent player master from current player temple : {score_distance_master_next_player}")
            print(f"Score total : {score}")

        return score

    #Retourne le contexte d'évaluation (joueurs, pièces, temples) selon le point de vue.
    #Utilisé par toutes les variantes d'heuristiques.
    def _get_evaluation_context(self, from_current_player_point_of_view:bool=True):
        """
        Retourne le contexte d'évaluation (joueurs, pièces, temples) selon le point de vue.
        Utilisé par toutes les variantes d'heuristiques.
        """
        p1 = self.current_player
        p2 = PLAYER_TWO_POSITION if p1 == PLAYER_ONE_POSITION else PLAYER_ONE_POSITION

        p1_student = PLAYER_ONE_STUDENT if p1 == PLAYER_ONE_POSITION else PLAYER_TWO_STUDENT
        p2_student = PLAYER_TWO_STUDENT if p1 == PLAYER_ONE_POSITION else PLAYER_ONE_STUDENT
        p1_master = PLAYER_ONE_MASTER if p1 == PLAYER_ONE_POSITION else PLAYER_TWO_MASTER
        p2_master = PLAYER_TWO_MASTER if p1 == PLAYER_ONE_POSITION else PLAYER_ONE_MASTER
        p1_temple = PLAYER_ONE_TEMPLE if p1 == PLAYER_ONE_POSITION else PLAYER_TWO_TEMPLE
        p2_temple = PLAYER_TWO_TEMPLE if p1 == PLAYER_ONE_POSITION else PLAYER_ONE_TEMPLE

        if from_current_player_point_of_view is False:
            p1, p2 = p2, p1
            p1_student, p2_student = p2_student, p1_student
            p1_master, p2_master = p2_master, p1_master
            p1_temple, p2_temple = p2_temple, p1_temple

        return {
            'p1': p1, 'p2': p2,
            'p1_student': p1_student, 'p2_student': p2_student,
            'p1_master': p1_master, 'p2_master': p2_master,
            'p1_temple': p1_temple, 'p2_temple': p2_temple
        }

    #Compte les pièces et retourne leurs positions.
    def _count_pieces(self, ctx):
        """Compte les pièces et retourne leurs positions."""
        p1_students = []
        p2_students = []
        p1_master_pos = None
        p2_master_pos = None

        for col in range(5):
            for row in range(5):
                cell = self.board[col][row]
                if cell == ctx['p1_student']:
                    p1_students.append((col, row))
                elif cell == ctx['p2_student']:
                    p2_students.append((col, row))
                elif cell == ctx['p1_master']:
                    p1_master_pos = (col, row)
                elif cell == ctx['p2_master']:
                    p2_master_pos = (col, row)

        return p1_students, p2_students, p1_master_pos, p2_master_pos

    # Heuristique agressive : priorité aux captures et au contrôle du centre.
    # - Bonus important pour capturer des pièces adverses
    # - Bonus pour contrôle du centre (cases autour de (2,2))
    # - Accepte plus facilement les sacrifices
    def heuristic_aggressive(self, from_current_player_point_of_view:bool=True, verbose:bool=False):
        """
        Heuristique agressive : priorité aux captures et au contrôle du centre.
        - Bonus important pour capturer des pièces adverses
        - Bonus pour contrôle du centre (cases autour de (2,2))
        - Accepte plus facilement les sacrifices
        """
        ctx = self._get_evaluation_context(from_current_player_point_of_view)

        # Victoire/Défaite
        game_ended, winner = self.game_has_ended()
        if game_ended:
            return 1000 if winner == ctx['p1'] else -1000

        p1_students, p2_students, p1_master_pos, p2_master_pos = self._count_pieces(ctx)

        # Différence d'étudiants (bonus augmenté pour captures)
        score_captures = (len(p1_students) - len(p2_students)) * 150

        # Contrôle du centre (cases (1,1) à (3,3))
        center_control = 0
        center_squares = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
        for col, row in center_squares:
            cell = self.board[col][row]
            if cell in [ctx['p1_student'], ctx['p1_master']]:
                center_control += 15
            elif cell in [ctx['p2_student'], ctx['p2_master']]:
                center_control -= 15

        # Distance agressive du maître vers temple adverse
        if p1_master_pos:
            temple_col, temple_row = ctx['p2_temple']
            distance = abs(p1_master_pos[0] - temple_col) + abs(p1_master_pos[1] - temple_row)
            score_attack = (8 - distance) * 25
        else:
            score_attack = 0

        # Pénalité réduite pour maître adverse proche (on accepte le risque)
        if p2_master_pos:
            temple_col, temple_row = ctx['p1_temple']
            distance = abs(p2_master_pos[0] - temple_col) + abs(p2_master_pos[1] - temple_row)
            score_defense = (8 - distance) * -10  # Pénalité réduite
        else:
            score_defense = 0

        score = score_captures + center_control + score_attack + score_defense

        if verbose:
            print(f"[Aggressive] Captures: {score_captures}, Centre: {center_control}, Attack: {score_attack}, Defense: {score_defense}, Total: {score}")

        return score

    #Heuristique défensive : protection du maître et structure solide.
    #- Priorité à bloquer l'adversaire
    # - Bonus pour étudiants protégeant le maître
    # - Malus important si le maître est exposé
    def heuristic_defensive(self, from_current_player_point_of_view:bool=True, verbose:bool=False):
        """
        Heuristique défensive : protection du maître et structure solide.
        - Priorité à bloquer l'adversaire
        - Bonus pour étudiants protégeant le maître
        - Malus important si le maître est exposé
        """
        ctx = self._get_evaluation_context(from_current_player_point_of_view)

        # Victoire/Défaite
        game_ended, winner = self.game_has_ended()
        if game_ended:
            return 1000 if winner == ctx['p1'] else -1000

        p1_students, p2_students, p1_master_pos, p2_master_pos = self._count_pieces(ctx)

        # Différence d'étudiants
        score_students = (len(p1_students) - len(p2_students)) * 100

        # Protection du maître : compter les étudiants adjacents
        protection_score = 0
        if p1_master_pos:
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                adj_col, adj_row = p1_master_pos[0] + dx, p1_master_pos[1] + dy
                if 0 <= adj_col < 5 and 0 <= adj_row < 5:
                    if self.board[adj_col][adj_row] == ctx['p1_student']:
                        protection_score += 25

        # Malus si maître exposé (peu de protection)
        if p1_master_pos and protection_score < 25:
            protection_score -= 30

        # Distance du maître adverse par rapport à notre temple (très importante)
        if p2_master_pos:
            temple_col, temple_row = ctx['p1_temple']
            distance = abs(p2_master_pos[0] - temple_col) + abs(p2_master_pos[1] - temple_row)
            score_threat = (8 - distance) * -30  # Pénalité forte
        else:
            score_threat = 0

        # Notre maître vers temple adverse (importance réduite)
        if p1_master_pos:
            temple_col, temple_row = ctx['p2_temple']
            distance = abs(p1_master_pos[0] - temple_col) + abs(p1_master_pos[1] - temple_row)
            score_advance = (8 - distance) * 10  # Bonus réduit
        else:
            score_advance = 0

        # Bonus pour garder le maître près de notre temple
        if p1_master_pos:
            temple_col, temple_row = ctx['p1_temple']
            distance_home = abs(p1_master_pos[0] - temple_col) + abs(p1_master_pos[1] - temple_row)
            score_home = (4 - distance_home) * 10 if distance_home <= 4 else 0
        else:
            score_home = 0

        score = score_students + protection_score + score_threat + score_advance + score_home

        if verbose:
            print(f"[Defensive] Students: {score_students}, Protection: {protection_score}, Threat: {score_threat}, Advance: {score_advance}, Home: {score_home}, Total: {score}")

        return score


    # Heuristique basée sur la mobilité et le contrôle de l'espace.
    # - Nombre de coups légaux disponibles
    # - Contrôle des cases centrales
    #  - Avancement des pièces
    def heuristic_mobility(self, from_current_player_point_of_view:bool=True, verbose:bool=False):
        ctx = self._get_evaluation_context(from_current_player_point_of_view)

        # Victoire/Défaite
        game_ended, winner = self.game_has_ended()
        if game_ended:
            return 1000 if winner == ctx['p1'] else -1000

        p1_students, p2_students, p1_master_pos, p2_master_pos = self._count_pieces(ctx)

        # Différence d'étudiants
        score_students = (len(p1_students) - len(p2_students)) * 100

        # Mobilité : nombre de coups disponibles
        num_moves = len(self.get_available_moves())
        score_mobility = num_moves * 5

        # Avancement des pièces (bonus pour pièces avancées vers l'adversaire)
        # Row 0 = notre côté, Row 4 = côté adverse
        advancement_score = 0
        for col, row in p1_students:
            advancement_score += row * 8  # Plus avancé = plus de points
        if p1_master_pos:
            advancement_score += p1_master_pos[1] * 5

        # Pénalité pour pièces adverses avancées
        for col, row in p2_students:
            advancement_score -= (4 - row) * 8
        if p2_master_pos:
            advancement_score -= (4 - p2_master_pos[1]) * 5

        # Distance maître vers temple
        if p1_master_pos:
            temple_col, temple_row = ctx['p2_temple']
            distance = abs(p1_master_pos[0] - temple_col) + abs(p1_master_pos[1] - temple_row)
            score_master = (8 - distance) * 15
        else:
            score_master = 0

        if p2_master_pos:
            temple_col, temple_row = ctx['p1_temple']
            distance = abs(p2_master_pos[0] - temple_col) + abs(p2_master_pos[1] - temple_row)
            score_master -= (8 - distance) * 15

        score = score_students + score_mobility + advancement_score + score_master

        if verbose:
            print(f"[Mobility] Students: {score_students}, Mobility: {score_mobility}, Advancement: {advancement_score}, Master: {score_master}, Total: {score}")

        return score


    #Heuristique positionnelle : structure de pions et contrôle territorial.
    # - Formation groupée des pièces
    # - Contrôle de colonnes/rangées clés
    # - Occupation des diagonales
    def heuristic_positional(self, from_current_player_point_of_view:bool=True, verbose:bool=False):
        
        ctx = self._get_evaluation_context(from_current_player_point_of_view)

        # Victoire/Défaite
        game_ended, winner = self.game_has_ended()
        if game_ended:
            return 1000 if winner == ctx['p1'] else -1000

        p1_students, p2_students, p1_master_pos, p2_master_pos = self._count_pieces(ctx)

        # Différence d'étudiants
        score_students = (len(p1_students) - len(p2_students)) * 100

        # Cohésion des pièces : bonus si les pièces sont proches les unes des autres
        cohesion_score = 0
        p1_pieces = p1_students + ([p1_master_pos] if p1_master_pos else [])
        for i, pos1 in enumerate(p1_pieces):
            for pos2 in p1_pieces[i+1:]:
                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                if distance <= 2:
                    cohesion_score += 10
                elif distance <= 3:
                    cohesion_score += 5

        # Contrôle de la colonne centrale (colonne 2)
        central_column = 0
        for row in range(5):
            cell = self.board[2][row]
            if cell in [ctx['p1_student'], ctx['p1_master']]:
                central_column += 12
            elif cell in [ctx['p2_student'], ctx['p2_master']]:
                central_column -= 12

        # Contrôle des rangées avancées
        territory_score = 0
        for col in range(5):
            for row in range(3, 5):  # Rangées 3 et 4 (côté adverse)
                cell = self.board[col][row]
                if cell in [ctx['p1_student'], ctx['p1_master']]:
                    territory_score += 15
            for row in range(0, 2):  # Rangées 0 et 1 (notre côté)
                cell = self.board[col][row]
                if cell in [ctx['p2_student'], ctx['p2_master']]:
                    territory_score -= 15

        # Distance maître vers temple
        if p1_master_pos:
            temple_col, temple_row = ctx['p2_temple']
            distance = abs(p1_master_pos[0] - temple_col) + abs(p1_master_pos[1] - temple_row)
            score_master = (8 - distance) * 15
        else:
            score_master = 0

        if p2_master_pos:
            temple_col, temple_row = ctx['p1_temple']
            distance = abs(p2_master_pos[0] - temple_col) + abs(p2_master_pos[1] - temple_row)
            score_master -= (8 - distance) * 15

        score = score_students + cohesion_score + central_column + territory_score + score_master

        if verbose:
            print(f"[Positional] Students: {score_students}, Cohesion: {cohesion_score}, Central: {central_column}, Territory: {territory_score}, Master: {score_master}, Total: {score}")

        return score

    #Heuristique avec bruit aléatoire pour générer des parties variées.
    #Utile pour créer des données d'entraînement diversifiées.
    def heuristic_noisy(self, from_current_player_point_of_view:bool=True, noise_factor:float=0.15, verbose:bool=False):
        """
        Heuristique avec bruit aléatoire pour générer des parties variées.
        Utile pour créer des données d'entraînement diversifiées.
        """
        base_score = self.heuristic_evaluation(from_current_player_point_of_view, verbose=False)

        # Ajouter du bruit proportionnel au score (évite le bruit sur les positions gagnantes/perdantes)
        if abs(base_score) < 900:  # Pas de bruit sur victoire/défaite
            noise = random.gauss(0, abs(base_score) * noise_factor + 20)
            score = base_score + noise
        else:
            score = base_score

        if verbose:
            print(f"[Noisy] Base: {base_score}, Final: {score}")

        return score


    def heuristic_weighted(self, weights:dict=None, from_current_player_point_of_view:bool=True, verbose:bool=False):
        """
        Heuristique configurable avec poids personnalisables.
        Permet de créer facilement de nombreuses variantes.

        Poids par défaut (équivalent à heuristic_evaluation):
        - student_diff: 100
        - master_advance: 20
        - master_threat: 20
        - center_control: 0
        - mobility: 0
        - cohesion: 0
        - protection: 0
        - advancement: 0
        """
        default_weights = {
            'student_diff': 100,
            'master_advance': 20,
            'master_threat': 20,
            'center_control': 0,
            'mobility': 0,
            'cohesion': 0,
            'protection': 0,
            'advancement': 0
        }

        if weights:
            default_weights.update(weights)
        w = default_weights

        ctx = self._get_evaluation_context(from_current_player_point_of_view)

        # Victoire/Défaite
        game_ended, winner = self.game_has_ended()
        if game_ended:
            return 1000 if winner == ctx['p1'] else -1000

        p1_students, p2_students, p1_master_pos, p2_master_pos = self._count_pieces(ctx)
        scores = {}

        # Différence d'étudiants
        scores['student_diff'] = (len(p1_students) - len(p2_students)) * w['student_diff']

        # Avance du maître vers temple adverse
        if p1_master_pos and w['master_advance'] != 0:
            temple_col, temple_row = ctx['p2_temple']
            distance = abs(p1_master_pos[0] - temple_col) + abs(p1_master_pos[1] - temple_row)
            scores['master_advance'] = (8 - distance) * w['master_advance']
        else:
            scores['master_advance'] = 0

        # Menace du maître adverse
        if p2_master_pos and w['master_threat'] != 0:
            temple_col, temple_row = ctx['p1_temple']
            distance = abs(p2_master_pos[0] - temple_col) + abs(p2_master_pos[1] - temple_row)
            scores['master_threat'] = (8 - distance) * -w['master_threat']
        else:
            scores['master_threat'] = 0

        # Contrôle du centre
        if w['center_control'] != 0:
            center_squares = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
            center = 0
            for col, row in center_squares:
                cell = self.board[col][row]
                if cell in [ctx['p1_student'], ctx['p1_master']]:
                    center += 1
                elif cell in [ctx['p2_student'], ctx['p2_master']]:
                    center -= 1
            scores['center_control'] = center * w['center_control']
        else:
            scores['center_control'] = 0

        # Mobilité
        if w['mobility'] != 0:
            scores['mobility'] = len(self.get_available_moves()) * w['mobility']
        else:
            scores['mobility'] = 0

        # Cohésion
        if w['cohesion'] != 0:
            p1_pieces = p1_students + ([p1_master_pos] if p1_master_pos else [])
            cohesion = 0
            for i, pos1 in enumerate(p1_pieces):
                for pos2 in p1_pieces[i+1:]:
                    distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                    if distance <= 2:
                        cohesion += 1
            scores['cohesion'] = cohesion * w['cohesion']
        else:
            scores['cohesion'] = 0

        # Protection du maître
        if w['protection'] != 0 and p1_master_pos:
            protection = 0
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                adj_col, adj_row = p1_master_pos[0] + dx, p1_master_pos[1] + dy
                if 0 <= adj_col < 5 and 0 <= adj_row < 5:
                    if self.board[adj_col][adj_row] == ctx['p1_student']:
                        protection += 1
            scores['protection'] = protection * w['protection']
        else:
            scores['protection'] = 0

        # Avancement des pièces
        if w['advancement'] != 0:
            adv = 0
            for col, row in p1_students:
                adv += row
            if p1_master_pos:
                adv += p1_master_pos[1]
            for col, row in p2_students:
                adv -= (4 - row)
            if p2_master_pos:
                adv -= (4 - p2_master_pos[1])
            scores['advancement'] = adv * w['advancement']
        else:
            scores['advancement'] = 0

        total = sum(scores.values())

        if verbose:
            print(f"[Weighted] {scores}, Total: {total}")

        return total


    """
    Retourne la matrice représentant l'état du plateau
    """
    def get_state(self):
        """
        Permet de convertir l'état en une matrice 5x5x10, plateau tourné du point de vue du joueur courant
        Plan 0 : positions des pions du joueur courant (0/1)
        Plan 1 : position du maître du joueur courant (0/1)
        Plan 2 : positions des pions de l'adversaire (0/1)
        Plan 3 : position du maître de l'adversaire (0/1)
        Plan 4 : Mouvements de la carte 1 du joueur courant (0/1) - pièce au centre en (2,2)
        Plan 5 : Mouvements de la carte 2 du joueur courant (0/1) - pièce au centre en (2,2)
        Plan 6 : Mouvements de la carte 1 de l'adversaire (0/1) - pièce au centre en (2,2)
        Plan 7 : Mouvements de la carte 2 de l'adversaire (0/1) - pièce au centre en (2,2)
        Plan 8 : Mouvements de la carte neutre (0/1) _ pièce au centre en (2,2)
        Plan 9 : matrix de 1 si le joueur courant est le premier joueur, -1 sinon
        """

        #Variables adaptées au point de vue
        student_current = PLAYER_ONE_STUDENT if self.current_player == PLAYER_ONE_POSITION else PLAYER_TWO_STUDENT
        student_next = PLAYER_TWO_STUDENT if self.current_player == PLAYER_ONE_POSITION else PLAYER_ONE_STUDENT
        master_current = PLAYER_ONE_MASTER if self.current_player == PLAYER_ONE_POSITION else PLAYER_TWO_MASTER
        master_next = PLAYER_TWO_MASTER if self.current_player == PLAYER_ONE_POSITION else PLAYER_ONE_MASTER

        channels = []
        board_array = np.array(self.board)

        #Plan 0 : positions des pions du joueur courant (0/1)
        channels.append((board_array == student_current).astype(float)) 

        #Plan 1 : position du maître du joueur courant (0/1)
        channels.append((board_array == master_current).astype(float))

        #Plan 2 : positions des pions de l'adversaire (0/1)
        channels.append((board_array == student_next).astype(float))

        #Plan 3 : position du maître de l'adversaire (0/1)
        channels.append((board_array == master_next).astype(float)) 

        #Plan 4
        channels.append(self.current_player_cards[0].getMatrix())

        #Plan 5
        channels.append(self.current_player_cards[1].getMatrix())

        #Plan 6
        channels.append(self.next_player_cards[0].getMatrix())

        #Plan 7
        channels.append(self.next_player_cards[1].getMatrix())

        #Plan 8
        channels.append(self.neutral_card.getMatrix())

        #Plan 9
        if self.first_player == self.current_player:
            channels.append(np.ones((5,5)))
        else:
            channels.append(-np.ones((5,5)))

        return channels

    #------------------------------------------------------------------------------------------------------------------------------------

    #Surcharge print(instance)
    def __str__(self):
        res_str = '**********************************************************************\n'
        if self.current_player == PLAYER_ONE_POSITION:
            res_str += f"Current player : X\n\n"
        else:
            res_str += "Current player : O\n\n"
        res_str = res_str+'\n'.join(
            f"{ row + 1} | " + ' '.join(self.cell_to_string(self.board[col][row]) for col in range(5))
            for row in range(5)
        )
        letters = [chr(i) for i in range(ord('A'), ord('A') + 5)]
        res_str += '\n'
        res_str += '  | ' + ' '.join(letters) + '\n'

        res_str += "\nNext player cards: "
        for card in self.next_player_cards:
            res_str += card.name+" "

        res_str += f"\nNeutral card: {self.neutral_card.name}"

        res_str += "\nCurrent player cards: \n"
        for card in self.current_player_cards:
            res_str += f"{card.print_value}\n"


        return f"{res_str}"

        return res_str

    #Représentation (pour machine learning)
    def __repr__(self):
        board_repr = f"{self.current_player}"
        for col in range(5):
            board_repr += '|' + ''.join([str(plr_type) for plr_type in self.board[col]])
        return board_repr



    