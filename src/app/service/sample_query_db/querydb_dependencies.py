from fastapi import HTTPException
from fastapi import Depends, Request
from sqlalchemy import Table, MetaData
from sqlalchemy import text

# Import local library
from .querydb_database import connect_DB
from .querydb_config import map_db
from .querydb_models import getModel

async def validate(request: Request):
    path_param = request.path_params.get("product")
    if path_param is None or path_param not in map_db.keys():
        raise HTTPException(status_code=400, detail="Invalid input")


async def get_db(request: Request, map_db=map_db, validate=Depends(validate)):
    product = map_db.get(request.path_params.get("product"))
    server = product.get("server")
    database = product.get("database")
    SessionLocal = connect_DB(server, database)
    map_db = SessionLocal()
    try:
        yield map_db
    finally:
        map_db.close()
