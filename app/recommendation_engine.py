def generate_recommendations(
    rainfall_risk,
    soil_risk,
    vegetation_risk,
    yield_risk,
    features
):

    recommendations = []

    # --------------------------------------------------
    # Rainfall
    # --------------------------------------------------

    if rainfall_risk == "High":

        recommendations.append(
            "Rainfall conditions appear limited. "
            "Monitor soil moisture closely and consider "
            "appropriate moisture-conservation practices."
        )

    elif rainfall_risk == "Moderate":

        recommendations.append(
            "Monitor rainfall closely during crop establishment "
            "and maintain appropriate soil-moisture management."
        )

    # --------------------------------------------------
    # Soil
    # --------------------------------------------------

    if soil_risk == "High":

        recommendations.append(
            "Soil conditions indicate elevated risk. "
            "Consider soil testing and locally appropriate "
            "soil fertility or amendment practices."
        )

    elif soil_risk == "Moderate":

        recommendations.append(
            "Consider soil testing and maintain appropriate "
            "fertility management."
        )

    # --------------------------------------------------
    # Vegetation
    # --------------------------------------------------

    if vegetation_risk == "High":

        recommendations.append(
            "Pre-plant vegetation indicators are relatively low. "
            "Monitor field conditions and crop establishment closely."
        )

    elif vegetation_risk == "Moderate":

        recommendations.append(
            "Vegetation indicators are moderate. "
            "Monitor crop establishment after planting."
        )

    # --------------------------------------------------
    # Yield
    # --------------------------------------------------

    if yield_risk == "High":

        recommendations.append(
            "Predicted yield is substantially below the "
            "historical crop reference. Review planting, "
            "soil and moisture-management conditions."
        )

    elif yield_risk == "Moderate":

        recommendations.append(
            "Predicted yield is below the historical crop "
            "reference. Pay particular attention to crop "
            "establishment and field management."
        )

    # --------------------------------------------------
    # Default
    # --------------------------------------------------

    if not recommendations:

        recommendations.append(
            "Current indicators are relatively favorable. "
            "Continue normal crop monitoring and management."
        )

    return recommendations
