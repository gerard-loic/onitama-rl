from fastapi import FastAPI
from pydantic import BaseModel
from requestmodels import PostGameRequest, PostGamePlayRequest
from gamemanager import GameManager
from sessionmemory import SessionMemory
import uvicorn

SessionMemory.init()

app = FastAPI()

# POST /game : débuter une nouvelle partie
# GET /game : obtenir l'état d'un jeu en cours
# POST /game/play : jouer un coup
# GET /players : obtenir les joueurs disponibles
# GET /cards : obtenir la liste des cartes


@app.post("/game")
def post_game(request: PostGameRequest):
    game = GameManager.create(player=request.player)

    return game.get_game_representation()

@app.get("/game/{game_uid}")
def get_game(game_uid: str):
    game = GameManager(uid=game_uid)

    return game.get_game_representation()

@app.post("/game/{game_uid}/play")
def post_game_play(request:PostGamePlayRequest, game_uid:str):
    game = GameManager(uid=game_uid)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)