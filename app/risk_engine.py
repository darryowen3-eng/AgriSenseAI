def classify_risk(value, low, high):

    if value < low:
        return "High"

    if value < high:
        return "Moderate"

    return "Low"


def calculate_rainfall_risk(features):

    value = features.get(
        "rainfall_planting_month"
    )

    if value is None:
        return "Unknown"

    return classify_risk(
        value,
        low=50,
        high=100
    )


def calculate_soil_risk(features):

    ph = features.get("soil_ph")

    if ph is None:
        return "Unknown"

    if ph < 5.0:
        return "High"

    if ph < 5.5:
        return "Moderate"

    return "Low"


def calculate_vegetation_risk(features):

    ndvi = features.get(
        "ndvi_preplant_mean"
    )

    if ndvi is None:
        return "Unknown"

    if ndvi < 0.25:
        return "High"

    if ndvi < 0.40:
        return "Moderate"

    return "Low"


def calculate_yield_risk(
    predicted_yield,
    crop_reference_yield
):

    if crop_reference_yield <= 0:
        return "Unknown"

    ratio = (
        predicted_yield /
        crop_reference_yield
    )

    if ratio < 0.60:
        return "High"

    if ratio < 0.85:
        return "Moderate"

    return "Low"


def calculate_overall_risk(
    rainfall_risk,
    soil_risk,
    vegetation_risk,
    yield_risk
):

    risks = [
        rainfall_risk,
        soil_risk,
        vegetation_risk,
        yield_risk
    ]

    scores = {
        "Low": 0,
        "Moderate": 1,
        "High": 2,
        "Unknown": 0
    }

    total = sum(
        scores.get(risk, 0)
        for risk in risks
    )

    if total >= 6:
        return "High"

    if total >= 3:
        return "Moderate"

    return "Low"
