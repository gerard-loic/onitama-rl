from pydantic import BaseModel

class PostGameRequest(BaseModel):
    player:str

class PostGamePlayRequest(BaseModel):
    from_pos_col:int
    from_pos_row:int
    to_pos_col:int
    to_pos_row:int
    card_idx:int