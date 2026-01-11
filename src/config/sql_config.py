from typing import Final
import pymysql

from enum import Enum
from pydantic_settings import BaseSettings


class CursorEnum(str, Enum):
    Dict = "Dict"

    @property
    def cursorclass(self):
        if self is CursorEnum.Dict:
            return pymysql.cursors.DictCursor
        else:
            raise "Not implemented yet"


class SQLConfig(BaseSettings):

    user: str
    password: str
    database: str
    cursor: CursorEnum

    @property
    def cursorclass(self) -> pymysql.cursors.Cursor:
        return self.cursor.cursorclass

    class Config:
        env_file = ".env"


sql_settings: Final[SQLConfig] = SQLConfig()