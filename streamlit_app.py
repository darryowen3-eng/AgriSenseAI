import streamlit as st
import requests
import pandas as pd
import numpy as np
import pydeck as pdk
from pathlib import Path
from datetime import date


# ============================================================
# CONFIG
# ============================================================

API_URL = "https://agrisense-fyvu.onrender.com"

BASE_DIR = Path(__file__).resolve().parent

COORDINATE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "zamstats_locations_coordinates.csv"
)

MASTER_FILE = (
    BASE_DIR
    / "data"
    / "master"
    / "agrisense_master_v5.csv"
)


st.set_page_config(
    page_title="AgriSense AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9f6;
    }

    .hero {
        padding: 1.5rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        background: linear-gradient(
            135deg,
            #12372a 0%,
            #1f5c42 55%,
            #2e7d5b 100%
        );
        color: white;
    }

    .hero h1 {
        font-size: 2.8rem;
        margin-bottom: 0.3rem;
    }

    .hero p {
        font-size: 1.15rem;
        opacity: 0.92;
    }

    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid #e4e9e4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }

    .metric-title {
        font-size: 0.9rem;
        color: #667066;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    .risk-card {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid #e4e9e4;
        text-align: center;
        min-height: 120px;
    }

    .risk-title {
        font-size: 0.9rem;
        color: #667066;
    }

    .risk-value {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }

    .recommendation {
        background: white;
        padding: 1rem 1.2rem;
        border-left: 5px solid #2e7d5b;
        border-radius: 8px;
        margin-bottom: 0.7rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .small-note {
        color: #6b746b;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPERS
# ============================================================

def api_get(endpoint):
    """GET request to the AgriSense API."""

    response = requests.get(
        f"{API_URL}{endpoint}",
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def api_post(endpoint, payload):
    """POST request to the AgriSense API."""

    response = requests.post(
        f"{API_URL}{endpoint}",
        json=payload,
        timeout=90
    )

    response.raise_for_status()

    return response.json()


def extract_list(data, possible_keys):
    """
    Handle APIs that return either:
        ["A", "B"]
    or:
        {"crops": ["A", "B"]}
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in possible_keys:

            if key in data and isinstance(data[key], list):
                return data[key]

    return []


def risk_icon(level):

    if not level:
        return "⚪"

    level = str(level).lower()

    if level == "low":
        return "🟢"

    if level == "moderate":
        return "🟡"

    if level in ["elevated", "medium"]:
        return "🟠"

    if level == "high":
        return "🔴"

    return "⚪"


def risk_description(level):

    if not level:
        return "No risk information available."

    level = str(level).lower()

    descriptions = {
        "low": "Conditions are relatively favorable.",
        "moderate": "Some conditions require monitoring.",
        "elevated": "Several conditions require attention.",
        "high": "Conditions indicate substantial agricultural risk."
    }

    return descriptions.get(
        level,
        "Monitor conditions closely."
    )


# ============================================================
# LOAD API DATA
# ============================================================

@st.cache_data(ttl=3600)
def load_crops():

    data = api_get("/crops")

    return extract_list(
        data,
        ["crops", "data", "items"]
    )


@st.cache_data(ttl=3600)
def load_locations():

    data = api_get("/locations")

    return extract_list(
        data,
        ["locations", "districts", "data", "items"]
    )


# ============================================================
# LOAD MAP DATA
# ============================================================

@st.cache_data
def load_map_data():

    if not COORDINATE_FILE.exists():

        return pd.DataFrame()

    locations = pd.read_csv(
        COORDINATE_FILE
    )

    locations["latitude"] = pd.to_numeric(
        locations["latitude"],
        errors="coerce"
    )

    locations["longitude"] = pd.to_numeric(
        locations["longitude"],
        errors="coerce"
    )

    locations = locations.dropna(
        subset=["latitude", "longitude"]
    )

    # One coordinate per province/district
    locations = (
        locations
        .sort_values(
            ["PROV", "DISTRICT"]
        )
        .groupby(
            ["PROV", "DISTRICT"],
            as_index=False
        )
        .first()
    )

    locations = locations.rename(
        columns={
            "PROV": "province",
            "DISTRICT": "district"
        }
    )

    # Historical yield
    if MASTER_FILE.exists():

        master = pd.read_csv(
            MASTER_FILE,
            usecols=[
                "province",
                "district",
                "yield_tonnes_per_ha"
            ]
        )

        historical = (
            master
            .groupby(
                ["province", "district"],
                as_index=False
            )
            ["yield_tonnes_per_ha"]
            .mean()
            .rename(
                columns={
                    "yield_tonnes_per_ha":
                    "historical_yield"
                }
            )
        )

        locations = locations.merge(
            historical,
            on=["province", "district"],
            how="left"
        )

    else:

        locations["historical_yield"] = np.nan

    return locations


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🌾 AgriSense AI</h1>

        <p>
        Intelligent Agricultural Risk, Yield Prediction
        & Decision Support System for Zambian Farmers
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API STATUS
# ============================================================

try:

    health = api_get("/health")

    api_online = True

except Exception:

    api_online = False


if api_online:

    st.success(
        "🟢 AgriSense AI decision engine is online."
    )

else:

    st.error(
        "🔴 AgriSense API is currently unavailable."
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌾 AgriSense AI")

st.sidebar.markdown(
    """
    ### Farm Assessment

    Enter four simple pieces of information.

    AgriSense combines them with agricultural
    environmental information to generate
    a yield and risk assessment.
    """
)

try:

    crops = load_crops()

except Exception as e:

    crops = []

    st.sidebar.error(
        f"Could not load crops: {e}"
    )


try:

    locations = load_locations()

except Exception as e:

    locations = []

    st.sidebar.error(
        f"Could not load locations: {e}"
    )


# ============================================================
# FARM INPUTS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("🌱 Farm Information")


if crops:

    crop = st.sidebar.selectbox(
        "Crop",
        sorted(crops)
    )

else:

    crop = st.sidebar.text_input(
        "Crop",
        value="Maize(for grain)"
    )


farm_size = st.sidebar.number_input(
    "Farm size (hectares)",
    min_value=0.01,
    max_value=10000.0,
    value=2.5,
    step=0.1
)


if locations:

    location = st.sidebar.selectbox(
        "District",
        sorted(locations)
    )

else:

    location = st.sidebar.text_input(
        "District",
        value="Mumbwa"
    )


planting_date = st.sidebar.date_input(
    "Planting date",
    value=date.today()
)


analyze = st.sidebar.button(
    "🚀 ANALYZE MY FARM",
    use_container_width=True
)


# ============================================================
# MAIN INTRO
# ============================================================

if not analyze and "prediction" not in st.session_state:

    st.markdown(
        """
        ## 🌱 Agricultural Decision Support

        AgriSense AI combines:

        - 🌧️ rainfall information
        - 🌱 soil characteristics
        - 🛰️ vegetation indicators
        - 🌾 crop information
        - 📍 geographic information
        - 📊 historical agricultural data

        to produce a farm-level yield and risk assessment.

        **Enter your farm information in the sidebar to begin.**
        """
    )


# ============================================================
# RUN PREDICTION
# ============================================================

if analyze:

    payload = {

        "crop": crop,

        "farm_size_ha": float(
            farm_size
        ),

        "location": location,

        "planting_date": planting_date.isoformat()
    }

    with st.spinner(
        "🌍 Analyzing agricultural conditions..."
    ):

        try:

            result = api_post(
                "/predict",
                payload
            )

            st.session_state.prediction = result

        except requests.exceptions.HTTPError as e:

            st.error(
                f"API returned an error: {e}"
            )

            try:

                st.code(
                    e.response.text
                )

            except Exception:
                pass

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# DISPLAY RESULT
# ============================================================

if "prediction" in st.session_state:

    result = st.session_state.prediction

    st.markdown(
        '<div class="section-title">🎯 Farm Prediction</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    predicted_yield = result.get(
        "predicted_yield_tonnes_per_ha",
        0
    )

    total_yield = result.get(
        "estimated_total_yield_tonnes",
        0
    )

    overall_risk = result.get(
        "risk_level",
        "Unknown"
    )

    crop_result = result.get(
        "crop",
        crop
    )

    location_result = result.get(
        "location",
        location
    )

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    Predicted Yield
                </div>

                <div class="metric-value">
                    {predicted_yield:.3f}
                    t/ha
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    Estimated Production
                </div>

                <div class="metric-value">
                    {total_yield:.3f}
                    tonnes
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    Crop
                </div>

                <div class="metric-value">
                    {crop_result}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    Location
                </div>

                <div class="metric-value">
                    {location_result}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # OVERALL RISK
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        f"{risk_icon(overall_risk)} Overall Agricultural Risk: "
        f"{overall_risk}"
    )

    st.info(
        risk_description(overall_risk)
    )


    # --------------------------------------------------------
    # RISK BREAKDOWN
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🌍 Risk Breakdown</div>',
        unsafe_allow_html=True
    )

    risk1, risk2, risk3 = st.columns(3)

    rainfall_risk = result.get(
        "rainfall_risk",
        "Unknown"
    )

    soil_risk = result.get(
        "soil_risk",
        "Unknown"
    )

    vegetation_risk = result.get(
        "vegetation_risk",
        "Unknown"
    )


    with risk1:

        st.markdown(
            f"""
            <div class="risk-card">

                <div class="risk-title">
                    🌧️ Rainfall
                </div>

                <div class="risk-value">
                    {risk_icon(rainfall_risk)}
                    {rainfall_risk}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with risk2:

        st.markdown(
            f"""
            <div class="risk-card">

                <div class="risk-title">
                    🌱 Soil
                </div>

                <div class="risk-value">
                    {risk_icon(soil_risk)}
                    {soil_risk}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with risk3:

        st.markdown(
            f"""
            <div class="risk-card">

                <div class="risk-title">
                    🛰️ Vegetation
                </div>

                <div class="risk-value">
                    {risk_icon(vegetation_risk)}
                    {vegetation_risk}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # ENVIRONMENTAL CONDITIONS
    # --------------------------------------------------------

    environmental = result.get(
        "environmental_features",
        {}
    )

    st.markdown(
        '<div class="section-title">🔬 Environmental Conditions</div>',
        unsafe_allow_html=True
    )

    if environmental:

        env_df = pd.DataFrame(
            {
                "Indicator": list(
                    environmental.keys()
                ),
                "Value": list(
                    environmental.values()
                )
            }
        )

        st.dataframe(
            env_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Environmental information was not returned by the API."
        )


    # --------------------------------------------------------
    # DECISION CONTEXT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🧠 Decision Context</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "These observations describe the environmental "
        "conditions used by the AgriSense decision engine. "
        "They are not independent causal explanations of "
        "the machine-learning prediction."
    )

    context_cols = st.columns(3)

    with context_cols[0]:

        st.markdown(
            f"""
            **🌧️ Rainfall**

            Risk level: **{rainfall_risk}**

            {risk_description(rainfall_risk)}
            """
        )

    with context_cols[1]:

        st.markdown(
            f"""
            **🌱 Soil**

            Risk level: **{soil_risk}**

            {risk_description(soil_risk)}
            """
        )

    with context_cols[2]:

        st.markdown(
            f"""
            **🛰️ Vegetation**

            Risk level: **{vegetation_risk}**

            {risk_description(vegetation_risk)}
            """
        )


    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">💡 Recommendations</div>',
        unsafe_allow_html=True
    )

    recommendations = result.get(
        "recommendations",
        []
    )

    if recommendations:

        for recommendation in recommendations:

            st.markdown(
                f"""
                <div class="recommendation">
                    ✅ {recommendation}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "No recommendations were returned."
        )


    # --------------------------------------------------------
    # HISTORICAL DISTRICT CONTEXT
    # --------------------------------------------------------

    map_data = load_map_data()

    if not map_data.empty:

        district_data = map_data[
            map_data["district"].astype(str).str.lower()
            ==
            str(location_result).lower()
        ]

        if not district_data.empty:

            historical_yield = district_data[
                "historical_yield"
            ].iloc[0]

            if pd.notna(historical_yield):

                st.markdown(
                    '<div class="section-title">'
                    '📊 Historical District Context'
                    '</div>',
                    unsafe_allow_html=True
                )

                difference = (
                    predicted_yield
                    -
                    historical_yield
                )

                h1, h2, h3 = st.columns(3)

                with h1:

                    st.metric(
                        "Predicted Yield",
                        f"{predicted_yield:.3f} t/ha"
                    )

                with h2:

                    st.metric(
                        "Historical District Mean",
                        f"{historical_yield:.3f} t/ha"
                    )

                with h3:

                    st.metric(
                        "Difference",
                        f"{difference:+.3f} t/ha"
                    )


    # --------------------------------------------------------
    # ZAMBIA MAP
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🗺️ Zambia Agricultural Map</div>',
        unsafe_allow_html=True
    )

    if not map_data.empty:

        map_display = map_data.copy()

        # Color points according to historical yield
        def yield_color(value):

            if pd.isna(value):

                return [150, 150, 150, 180]

            if value < 0.5:

                return [220, 53, 69, 190]

            elif value < 1.0:

                return [255, 193, 7, 190]

            elif value < 1.5:

                return [46, 125, 91, 190]

            else:

                return [25, 118, 210, 190]


        map_display["color"] = map_display[
            "historical_yield"
        ].apply(yield_color)


        # Highlight selected district
        map_display["selected"] = (
            map_display["district"].astype(str).str.lower()
            ==
            str(location_result).lower()
        )

        map_display.loc[
            map_display["selected"],
            "color"
        ] = map_display.loc[
            map_display["selected"],
            "selected"
        ].apply(
            lambda _: [255, 87, 34, 255]
        )


        layer = pdk.Layer(
            "ScatterplotLayer",

            data=map_display,

            get_position=[
                "longitude",
                "latitude"
            ],

            get_fill_color="color",

            get_radius=12000,

            pickable=True,

            auto_highlight=True
        )


        tooltip = {
            "html": """
            <b>{district}</b><br/>
            Province: {province}<br/>
            Historical mean yield:
            {historical_yield} t/ha
            """,

            "style": {
                "backgroundColor": "white",
                "color": "black"
            }
        }


        view_state = pdk.ViewState(
            latitude=-13.5,
            longitude=27.8,
            zoom=4.7,
            pitch=0
        )


        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style=None
        )


        st.pydeck_chart(
            deck,
            use_container_width=True
        )


        st.caption(
            "Map shows district-level historical mean yield "
            "from the AgriSense training dataset. "
            "The selected district is highlighted."
        )

    else:

        st.warning(
            "Map coordinate data is not available."
        )


    # --------------------------------------------------------
    # DOWNLOAD RESULT
    # --------------------------------------------------------

    st.markdown("---")

    result_download = pd.DataFrame(
        [
            {
                "crop": crop_result,
                "location": location_result,
                "planting_date": result.get(
                    "planting_date",
                    planting_date.isoformat()
                ),
                "farm_size_ha": farm_size,
                "predicted_yield_tonnes_per_ha":
                    predicted_yield,
                "estimated_total_yield_tonnes":
                    total_yield,
                "risk_level":
                    overall_risk,
                "rainfall_risk":
                    rainfall_risk,
                "soil_risk":
                    soil_risk,
                "vegetation_risk":
                    vegetation_risk
            }
        ]
    )

    st.download_button(
        "📥 Download Farm Assessment",
        data=result_download.to_csv(
            index=False
        ),
        file_name="agrisense_farm_assessment.csv",
        mime="text/csv"
    )
