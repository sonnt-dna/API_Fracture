
from fastapi import APIRouter
from fastapi import Depends, Request

# Import local library
from .querydb_dependencies import validate, get_db
from .querydb_handlers import get_data_filter
#from .querydb_config import config

querydb_router = APIRouter()

@querydb_router.get("/{product}/{tableName}")
async def get_columns(request: Request, validate=Depends(validate),
                   db=Depends(get_db)):
    return await get_data_filter(request=request, db=db)
