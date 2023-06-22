from fastapi import Request, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.types import Date, Integer, Float, BIGINT
from pydantic import BaseModel
from typing import List

# Import local library
from .querydb_models import getModel

class QueryResponse(BaseModel):
    data: List[dict]
    total_records: int
    records_returned: int

async def get_data_filter(request: Request, db: Session):
    try:
        haveLength = request.query_params._dict.get("length")
        length = int(request.query_params._dict.pop('length', 5000))
        orderby = request.query_params._dict.pop('orderby', None)
        offset = int(request.query_params._dict.pop('offset', 0))
        tableName = request.path_params.get("tableName")
        TableModel = getModel(tableName, db)
        condition = []
        str_sql = text("SELECT table_name FROM information_schema.tables")
        result_query = db.execute(str_sql)
        results = result_query.fetchall()
        table_names = [row[0] for row in results]
        drop = ["relationshipColumns", "relationships", "database_firewall_rules"]
        list_table = [elem for elem in table_names if elem not in drop]
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid input")
    if tableName is not None and tableName not in list_table:
        raise HTTPException(status_code=400, detail="Table not found. Please check your table name again")

    if orderby is not None and orderby not in TableModel.c.keys():
        raise HTTPException(status_code=400, detail=f"order by column {orderby} not found in table {tableName}")
    if offset != 0 and orderby is None:
        raise HTTPException(status_code=400, detail="order by is required when you want to use offset")
    for column in TableModel.columns:
        if isinstance(column.type, (Integer, Float, BIGINT)):
            min = request.query_params._dict.pop('Min_' + column.name, None)
            max = request.query_params._dict.pop('Max_' + column.name, None)
            if max:
                condition.append(TableModel.c[column.name] <= max)
            if min:
                condition.append(TableModel.c[column.name] >= min)
        if isinstance(column.type, (Date)):
            min = request.query_params._dict.pop('Start_' + column.name, None)
            max = request.query_params._dict.pop('End_' + column.name, None)
            if max:
                condition.append(TableModel.c[column.name] <= max)
            if min:
                condition.append(TableModel.c[column.name] > min)

    others = request.query_params._dict

    for key in others.keys():
        if key.rstrip('_MAX').rstrip('_MIN') not in TableModel.c.keys():
            raise HTTPException(status_code=400, detail=f"{key} not found in table {tableName}")

    for key, value in others.items():
        condition.append(TableModel.c[key] == value)
    total_records = db.query(TableModel).count()
    try:
        if offset and orderby:
            results = db.query(TableModel) \
                .filter(*condition) \
                .order_by(TableModel.c[orderby]) \
                .offset(offset) \
                .limit(length) \
                .all()
        else:
            results = db.query(TableModel) \
                .filter(*condition) \
                .limit(length) \
                .all()
        res = []
        for result in results:
            res.append(result._mapping)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid input")
    if len(res) == 0:
        raise HTTPException(status_code=404, detail="No records found with your input")
    responseContainer = {}
    if total_records > 5000 and length >= 5000 and not haveLength:
        responseContainer["warnings"] =  "The API can only return 5000 rows in JSON format.  Please consider constraining your request with parameters or using length, offset, orderby to paginate results."
    responseContainer["total_records"] = total_records
    responseContainer["records_returned"] = len(res)
    responseContainer["offset"] = offset
    responseContainer["orderby"] = "" if orderby is None else orderby
    responseContainer["data"] = res
    return responseContainer

