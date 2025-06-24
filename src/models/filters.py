from dataclasses import dataclass
from typing import List, Optional
from uuid import UUID

@dataclass
class Filter:
    column: str
    type: str  # eq, ne, gt, gte, lt, lte, like, in
    value: str

@dataclass
class OrderBy:
    column: str
    order: str  # asc, desc

@dataclass
class GetListFilter:
    page: int = 1
    limit: int = 10
    filters: Optional[List[Filter]] = None
    order_by: Optional[List[OrderBy]] = None
