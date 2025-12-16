from pydantic import BaseModel
from typing import Optional

class InputData(BaseModel):
    total_amount: float
    avg_amount: float
    txn_count: float
    std_amount: float
    ProductCategory: str
    ChannelId: str
    ProviderId: str
    PricingStrategy: str
