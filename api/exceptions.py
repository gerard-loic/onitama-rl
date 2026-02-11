#Base pour toutes les erreurs métier
class AppException(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code

class InvalidPlayerException(AppException):
    def __init__(self, player: str):
        super().__init__(f"Invalid action : it's not '{player}' turn", status_code=400)

class InvalidMoveException(AppException):
    def __init__(self):
        super().__init__(f"Invalid move", status_code=400)

class GameEndedException(AppException):
    def __init__(self):
        super().__init__(f"The game ended", status_code=400)

class InvalidSessionException(AppException):
    def __init__(self, session: str):
        super().__init__(f"Game session '{session}' does not exists", status_code=404)

class PlayerNotFoundException(AppException):
    def __init__(self, player: str):
        super().__init__(f"Player '{player}' does not exists", status_code=404)