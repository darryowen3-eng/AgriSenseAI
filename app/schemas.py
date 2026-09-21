from pydantic import BaseModel, Field


class FarmRequest(BaseModel):
    crop: str
    farm_size_ha: float = Field(gt=0)
    location: str
    planting_date: str


class PredictionResponse(BaseModel):
    crop: str
    location: str
    planting_date: str

    predicted_yield_tonnes_per_ha: float
    estimated_total_yield_tonnes: float

    risk_level: str
    rainfall_risk: str
    soil_risk: str
    vegetation_risk: str

    recommendations: list[str]

    environmental_features: dict
