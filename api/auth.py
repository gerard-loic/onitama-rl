import hashlib
import os
import secrets

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

security = HTTPBasic()


def _load_users() -> dict[str, str]:
    raw = os.getenv("USERS", "")
    users: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" not in entry:
            continue
        username, password = entry.split(":", 1)
        users[username] = hashlib.sha256(password.encode()).hexdigest()
    return users


_users = _load_users()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    password_hash = hashlib.sha256(credentials.password.encode()).hexdigest()
    expected_hash = _users.get(credentials.username)

    if expected_hash is None or not secrets.compare_digest(password_hash, expected_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
