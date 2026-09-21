import requests
import streamlit as st


API_URL = "https://agrisense-fyvu.onrender.com"


st.set_page_config(
    page_title="AgriSense AI",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 20px;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }

    .big-number {
        font-size: 40px;
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌾 AgriSense AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Intelligent Agricultural Risk, Yield Prediction
    & Decision Support System
    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🌱 Farm Information")

# ------------------------------------------------------------
# Fetch crops
# ------------------------------------------------------------

try:

    crop_response = requests.get(
        f"{API_URL}/crops",
        timeout=5
    )

    crops = crop_response.json()["crops"]

except Exception:

    st.error(
        "Could not connect to AgriSense API."
    )

    st.stop()


# ------------------------------------------------------------
# Fetch locations
# ------------------------------------------------------------

try:

    location_response = requests.get(
        f"{API_URL}/locations",
        timeout=5
    )

    locations = (
        location_response
        .json()["locations"]
    )

except Exception:

    st.error(
        "Could not load locations."
    )

    st.stop()


# ============================================================
# INPUTS
# ============================================================

crop = st.sidebar.selectbox(
    "Crop",
    crops
)

farm_size = st.sidebar.number_input(
    "Farm size (hectares)",
    min_value=0.1,
    value=2.0,
    step=0.1
)

location = st.sidebar.selectbox(
    "District",
    locations
)

planting_date = st.sidebar.date_input(
    "Planting date"
)


analyze = st.sidebar.button(
    "🔍 ANALYZE MY FARM",
    use_container_width=True
)


# ============================================================
# LANDING PAGE
# ============================================================

if not analyze:

    st.info(
        "Enter your farm information on the left "
        "and click ANALYZE MY FARM."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Historical Records",
            "31,928"
        )

    with col2:

        st.metric(
            "Environmental Sources",
            "3"
        )

    with col3:

        st.metric(
            "Farmer Inputs",
            "4"
        )

    st.divider()

    st.subheader(
        "How AgriSense AI works"
    )

    st.write(
        """
        AgriSense AI combines historical agricultural
        production records with rainfall, soil and
        vegetation indicators to estimate crop yield
        and identify agricultural risk.
        """
    )

    st.stop()


# ============================================================
# REQUEST
# ============================================================

payload = {

    "crop": crop,

    "farm_size_ha": farm_size,

    "location": location,

    "planting_date": str(
        planting_date
    )
}


with st.spinner(
    "Analyzing your farm..."
):

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=60
        )

    except Exception as e:

        st.error(
            f"API connection failed: {e}"
        )

        st.stop()


if response.status_code != 200:

    st.error(
        response.text
    )

    st.stop()


result = response.json()


# ============================================================
# RESULTS
# ============================================================

st.header(
    "🌾 Farm Assessment"
)


# ------------------------------------------------------------
# Main metrics
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Expected Yield",
        f"{result['predicted_yield_tonnes_per_ha']:.2f} t/ha"
    )


with col2:

    st.metric(
        "Estimated Production",
        f"{result['estimated_total_yield_tonnes']:.2f} tonnes"
    )


with col3:

    st.metric(
        "Overall Risk",
        result["risk_level"]
    )


st.divider()


# ============================================================
# RISK PANEL
# ============================================================

st.subheader(
    "⚠️ Agricultural Risk"
)


risk1, risk2, risk3, risk4 = st.columns(4)


with risk1:

    st.metric(
        "Overall",
        result["risk_level"]
    )


with risk2:

    st.metric(
        "🌧️ Rainfall",
        result["rainfall_risk"]
    )


with risk3:

    st.metric(
        "🌱 Soil",
        result["soil_risk"]
    )


with risk4:

    st.metric(
        "🛰️ Vegetation",
        result["vegetation_risk"]
    )


st.divider()


# ============================================================
# ENVIRONMENTAL DATA
# ============================================================

st.subheader(
    "🌍 Environmental Indicators"
)


environment = result[
    "environmental_features"
]


e1, e2, e3 = st.columns(3)


with e1:

    st.metric(
        "Rainfall",
        (
            f"{environment['rainfall_planting_month']:.1f} mm"
            if environment[
                "rainfall_planting_month"
            ] is not None
            else "N/A"
        )
    )


with e2:

    st.metric(
        "Soil pH",
        (
            f"{environment['soil_ph']:.2f}"
            if environment["soil_ph"]
            is not None
            else "N/A"
        )
    )


with e3:

    st.metric(
        "Pre-plant NDVI",
        (
            f"{environment['ndvi_preplant_mean']:.3f}"
            if environment[
                "ndvi_preplant_mean"
            ] is not None
            else "N/A"
        )
    )


st.divider()


# ============================================================
# RECOMMENDATIONS
# ============================================================

st.subheader(
    "🧠 AgriSense Recommendations"
)


for recommendation in result[
    "recommendations"
]:

    st.info(
        recommendation
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "View technical assessment"
):

    st.write(
        "Profile source:",
        environment.get(
            "profile_source"
        )
    )

    st.write(
        "Historical crop reference:",
        environment.get(
            "crop_reference_yield"
        ),
        "t/ha"
    )
