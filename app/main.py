from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    FarmRequest,
    PredictionResponse
)

from app.feature_builder import (
    feature_builder
)

from app.predictor import (
    predictor
)

from app.risk_engine import (
    calculate_rainfall_risk,
    calculate_soil_risk,
    calculate_vegetation_risk,
    calculate_yield_risk,
    calculate_overall_risk
)

from app.recommendation_engine import (
    generate_recommendations
)


BASE_DIR = Path(__file__).resolve().parent.parent

MASTER_PATH = (
    BASE_DIR /
    "data/master/agrisense_master_v5.csv"
)


app = FastAPI(
    title="AgriSense AI",
    description=(
        "Intelligent Agricultural Risk, "
        "Yield Prediction and Decision Support System"
    ),
    version="1.0.0"
)


# --------------------------------------------------
# Load reference data
# --------------------------------------------------

master = pd.read_csv(
    MASTER_PATH
)


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "AgriSense AI is running",
        "status": "online"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Available crops
# --------------------------------------------------

@app.get("/crops")
def get_crops():

    crops = sorted(
        master["crop"]
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "crops": crops
    }


# --------------------------------------------------
# Available districts
# --------------------------------------------------

@app.get("/locations")
def get_locations():

    locations = sorted(
        master["district"]
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "locations": locations
    }


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: FarmRequest):

    try:

        features, metadata = (
            feature_builder.build_features(
                crop=request.crop,
                farm_size_ha=request.farm_size_ha,
                location=request.location,
                planting_date=request.planting_date
            )
        )

        predicted_yield = predictor.predict(
            features
        )

        # Prevent impossible negative prediction
        predicted_yield = max(
            0.0,
            predicted_yield
        )

        total_yield = (
            predicted_yield *
            request.farm_size_ha
        )

        # --------------------------------------------------
        # Environmental feature dictionary
        # --------------------------------------------------

        feature_row = features.iloc[0]

        environmental = {}

        for column in features.columns:

            if column in [
                "crop",
                "province",
                "district",
                "farm_size_ha",
                "planting_month"
            ]:
                continue

            value = feature_row[column]

            if pd.isna(value):
                environmental[column] = None
            else:
                environmental[column] = float(value)

        # --------------------------------------------------
        # Risk
        # --------------------------------------------------

        rainfall_risk = (
            calculate_rainfall_risk(
                environmental
            )
        )

        soil_risk = (
            calculate_soil_risk(
                environmental
            )
        )

        vegetation_risk = (
            calculate_vegetation_risk(
                environmental
            )
        )

        # Historical crop reference
        crop_data = master[
            master["crop"] ==
            metadata["crop"]
        ]

        if len(crop_data) > 0:

            crop_reference = float(
                crop_data[
                    "yield_tonnes_per_ha"
                ].median()
            )

        else:

            crop_reference = float(
                master[
                    "yield_tonnes_per_ha"
                ].median()
            )

        yield_risk = calculate_yield_risk(
            predicted_yield,
            crop_reference
        )

        overall_risk = (
            calculate_overall_risk(
                rainfall_risk,
                soil_risk,
                vegetation_risk,
                yield_risk
            )
        )

        # --------------------------------------------------
        # Recommendations
        # --------------------------------------------------

        recommendations = (
            generate_recommendations(
                rainfall_risk=rainfall_risk,
                soil_risk=soil_risk,
                vegetation_risk=vegetation_risk,
                yield_risk=yield_risk,
                features=environmental
            )
        )

        return {
            "crop": metadata["crop"],
            "location": metadata["district"],
            "planting_date": request.planting_date,

            "predicted_yield_tonnes_per_ha": round(
                predicted_yield,
                3
            ),

            "estimated_total_yield_tonnes": round(
                total_yield,
                3
            ),

            "risk_level": overall_risk,

            "rainfall_risk": rainfall_risk,
            "soil_risk": soil_risk,
            "vegetation_risk": vegetation_risk,

            "recommendations": recommendations,

            "environmental_features": {
                **environmental,
                "profile_source": metadata[
                    "profile_source"
                ],
                "crop_reference_yield": round(
                    crop_reference,
                    3
                )
            }
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
