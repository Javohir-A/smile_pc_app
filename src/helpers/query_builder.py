from sqlalchemy.orm import Query
from sqlalchemy import asc, desc, or_, and_
from src.models.filters import GetListFilter, Filter, OrderBy
from typing import List

def apply_filters(query: Query, model, filters: List[Filter]) -> Query:
    for f in filters:
        col = getattr(model, f.column, None)
        if col is None:
            continue

        if f.type == "eq":
            query = query.filter(col == f.value)
        elif f.type == "ne":
            query = query.filter(col != f.value)
        elif f.type == "gt":
            query = query.filter(col > f.value)
        elif f.type == "gte":
            query = query.filter(col >= f.value)
        elif f.type == "lt":
            query = query.filter(col < f.value)
        elif f.type == "lte":
            query = query.filter(col <= f.value)
        elif f.type == "like":
            query = query.filter(col.like(f"%{f.value}%"))
        elif f.type == "in":
            query = query.filter(col.in_(f.value.split(",")))  # comma-separated
    return query

def apply_ordering(query: Query, model, orders: List[OrderBy]) -> Query:
    for o in orders:
        col = getattr(model, o.column, None)
        if col is None:
            continue

        if o.order.lower() == "asc":
            query = query.order_by(asc(col))
        elif o.order.lower() == "desc":
            query = query.order_by(desc(col))
    return query


def apply_pagination(query: Query, page: int=1, limit: int=10) -> Query:
    offset = (page - 1) * limit
    return query.offset(offset).limit(limit)
