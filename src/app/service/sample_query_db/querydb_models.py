from sqlalchemy import Table, MetaData
from sqlalchemy.orm import Session


def getModel(tableName: str, db: Session):
    metadata = MetaData()
    TableModel = Table(tableName, metadata, autoload_with=db.bind)
    return TableModel
