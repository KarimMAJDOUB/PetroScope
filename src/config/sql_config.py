import pymysql
import os
from enum import Enum
from pydantic_settings import BaseSettings
from typing import Final

import os

# Chemin absolu du .env
from pathlib import Path

# remonte deux niveaux depuis src/config/ pour atteindre la racine du projet
env_path = Path(__file__).parent.parent.parent / ".env"
print("Chargement du .env depuis :", env_path)

# Lire le fichier ligne par ligne
with open(env_path) as f:
    for line in f:
        if "=" in line:
            key, value = line.strip().split("=", 1)
            os.environ[key] = value


class CursorEnum(str, Enum):
    Dict = "Dict"

    @property
    def cursorclass(self):
        if self is CursorEnum.Dict:
            return pymysql.cursors.DictCursor
        else:
            raise NotImplementedError("Cursor non implémenté")

class SQLConfig(BaseSettings):
    
    user: str
    password: str
    database: str
    cursor: CursorEnum

    @property
    def cursorclass(self):
        return self.cursor.cursorclass

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
        env_file = None  # on a déjà injecté os.environ

sql_settings: Final[SQLConfig] = SQLConfig()
