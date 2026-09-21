import streamlit as st
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import pydeck as pdk


# ============================================================
# CONFIGURATION
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


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AgriSense AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# API FUNCTIONS
# ============================================================

def api_get(endpoint):
    response = requests.get(
        f"{API_URL}{endpoint}",
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def api_post(endpoint, payload):
    response = requests.post(
        f"{API_URL}{endpoint}",
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# DATA HELPERS
# ============================================================

def extract_list(data, keys):
    """
    Safely extract a list from different possible API formats.
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in keys:

            value = data.get(key)

            if isinstance(value, list):
                return value

    return []


def clean_list(values):
    """
    Convert API values into clean strings.
    """

    if not values:
        return []

    cleaned = []

    for value in values:

        if value is None:
            continue

        text = str(value).strip()

        if text:
            cleaned.append(text)

    return sorted(
        list(set(cleaned))
    )


def get_risk_icon(level):

    if level is None:
        return "⚪"

    level = str(level).strip().lower()

    if level == "low":
        return "🟢"

    if level == "moderate":
        return "🟡"

    if level == "elevated":
        return "🟠"

    if level == "high":
        return "🔴"

    return "⚪"


def get_risk_description(level):

    if level is None:
        return "Risk information is unavailable."

    level = str(level).strip().lower()

    descriptions = {

        "low":
            "Conditions are relatively favorable.",

        "moderate":
            "Some conditions require monitoring.",

        "elevated":
            "Several conditions require attention.",

        "high":
            "Conditions indicate elevated agricultural risk."
    }

    return descriptions.get(
        level,
        "Monitor conditions closely."
    )


def clean_recommendations(value):
    """
    Make recommendations display safely regardless
    of the API response format.
    """

    if value is None:
        return []

    if isinstance(value, str):

        text = value.strip()

        return [text] if text else []

    if isinstance(value, (list, tuple)):

        output = []

        for item in value:

            if item is None:
                continue

            text = str(item).strip()

            if text:
                output.append(text)

        return output

    return [str(value)]


# ============================================================
# API DATA
# ============================================================

@st.cache_data(ttl=3600)
def load_crops():

    try:

        data = api_get("/crops")

        return clean_list(
            extract_list(
                data,
                ["crops", "data", "items"]
            )
        )

    except Exception:

        return []


@st.cache_data(ttl=3600)
def load_locations():

    try:

        data = api_get("/locations")

        return clean_list(
            extract_list(
                data,
                [
                    "locations",
                    "districts",
                    "data",
                    "items"
                ]
            )
        )

    except Exception:

        return []


# ============================================================
# MAP DATA
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
        subset=[
            "latitude",
            "longitude"
        ]
    )

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

    # --------------------------------------------------------
    # Historical yield
    # --------------------------------------------------------

    if MASTER_FILE.exists():

        master = pd.read_csv(
            MASTER_FILE,
            usecols=[
                "crop",
                "province",
                "district",
                "yield_tonnes_per_ha"
            ]
        )

        historical = (
            master
            .groupby(
                [
                    "province",
                    "district"
                ],
                as_index=False
            )[
                "yield_tonnes_per_ha"
            ]
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
            on=[
                "province",
                "district"
            ],
            how="left"
        )

    else:

        locations["historical_yield"] = np.nan

    return locations


# ============================================================
# HEADER
# ============================================================

st.title("🌾 AgriSense AI")

st.subheader(
    "Intelligent Agricultural Risk, Yield Prediction "
    "& Decision Support System for Zambian Farmers"
)

st.caption(
    "Agricultural intelligence powered by rainfall, "
    "soil, satellite vegetation indicators and machine learning."
)


# ============================================================
# API STATUS
# ============================================================

try:

    api_get("/health")

    st.success(
        "🟢 AgriSense AI decision engine is online."
    )

except Exception:

    st.error(
        "🔴 AgriSense AI decision engine is currently unavailable."
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌾 Farm Assessment")

st.sidebar.write(
    "Enter four simple pieces of information "
    "to assess a farm."
)


# ============================================================
# LOAD INPUT OPTIONS
# ============================================================

crops = load_crops()

locations = load_locations()


# ============================================================
# FARM INPUTS
# ============================================================

if crops:

    crop = st.sidebar.selectbox(
        "Crop",
        crops
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
        locations
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


st.sidebar.markdown("---")


analyze = st.sidebar.button(
    "🚀 ANALYZE MY FARM",
    use_container_width=True
)


# ============================================================
# INITIAL SCREEN
# ============================================================

if (
    not analyze
    and "prediction" not in st.session_state
):

    st.header("🌱 Agricultural Decision Support")

    st.write(
        "AgriSense AI combines agricultural data, "
        "environmental indicators and machine learning "
        "to provide a farm-level assessment."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "🌧️ **Rainfall**\n\n"
            "Historical rainfall indicators."
        )

    with col2:

        st.info(
            "🌱 **Soil**\n\n"
            "Soil characteristics from SoilGrids."
        )

    with col3:

        st.info(
            "🛰️ **Vegetation**\n\n"
            "Satellite-derived NDVI indicators."
        )

    st.write("")

    st.write(
        "Use the controls on the left to run a farm assessment."
    )


# ============================================================
# RUN PREDICTION
# ============================================================

if analyze:

    payload = {

        "crop":
            crop,

        "farm_size_ha":
            float(farm_size),

        "location":
            location,

        "planting_date":
            planting_date.isoformat()
    }

    with st.spinner(
        "🌍 Analyzing farm conditions..."
    ):

        try:

            result = api_post(
                "/predict",
                payload
            )

            st.session_state.prediction = result

        except requests.exceptions.HTTPError as error:

            st.error(
                "The prediction API returned an error."
            )

            if error.response is not None:

                st.code(
                    error.response.text
                )

        except Exception as error:

            st.error(
                "Could not connect to the prediction service."
            )

            st.exception(error)


# ============================================================
# DISPLAY PREDICTION
# ============================================================

if "prediction" in st.session_state:

    result = st.session_state.prediction


    # ========================================================
    # MAIN PREDICTION
    # ========================================================

    st.header("🎯 Farm Prediction")


    predicted_yield = float(
        result.get(
            "predicted_yield_tonnes_per_ha",
            0
        )
    )


    total_yield = float(
        result.get(
            "estimated_total_yield_tonnes",
            predicted_yield * float(farm_size)
        )
    )


    result_crop = result.get(
        "crop",
        crop
    )


    result_location = result.get(
        "location",
        location
    )


    overall_risk = result.get(
        "risk_level",
        "Unknown"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Predicted Yield",
            f"{predicted_yield:.3f} t/ha"
        )


    with col2:

        st.metric(
            "Estimated Production",
            f"{total_yield:.3f} tonnes"
        )


    with col3:

        st.metric(
            "Crop",
            result_crop
        )


    with col4:

        st.metric(
            "Location",
            result_location
        )


    # ========================================================
    # OVERALL RISK
    # ========================================================

    st.divider()

    st.header(
        f"{get_risk_icon(overall_risk)} "
        f"Overall Agricultural Risk: {overall_risk}"
    )

    st.info(
        get_risk_description(
            overall_risk
        )
    )


    # ========================================================
    # RISK BREAKDOWN
    # ========================================================

    st.header("🌍 Risk Breakdown")


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


    risk1, risk2, risk3 = st.columns(3)


    with risk1:

        st.subheader("🌧️ Rainfall")

        st.metric(
            "Risk",
            f"{get_risk_icon(rainfall_risk)} "
            f"{rainfall_risk}"
        )

        st.caption(
            get_risk_description(
                rainfall_risk
            )
        )


    with risk2:

        st.subheader("🌱 Soil")

        st.metric(
            "Risk",
            f"{get_risk_icon(soil_risk)} "
            f"{soil_risk}"
        )

        st.caption(
            get_risk_description(
                soil_risk
            )
        )


    with risk3:

        st.subheader("🛰️ Vegetation")

        st.metric(
            "Risk",
            f"{get_risk_icon(vegetation_risk)} "
            f"{vegetation_risk}"
        )

        st.caption(
            get_risk_description(
                vegetation_risk
            )
        )


    # ========================================================
    # ENVIRONMENTAL FEATURES
    # ========================================================

    st.header(
        "🔬 Environmental Conditions"
    )


    environmental = result.get(
        "environmental_features",
        {}
    )


    if isinstance(
        environmental,
        dict
    ) and environmental:

        rows = []

        for key, value in environmental.items():

            if value is None:

                display_value = "Not available"

            elif isinstance(
                value,
                float
            ):

                display_value = round(
                    value,
                    3
                )

            else:

                display_value = value

            rows.append(
                {
                    "Indicator":
                        key.replace(
                            "_",
                            " "
                        ).title(),

                    "Value":
                        display_value
                }
            )


        environmental_df = pd.DataFrame(
            rows
        )


        st.dataframe(
            environmental_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Environmental feature details "
            "were not returned by the API."
        )


    # ========================================================
    # DECISION CONTEXT
    # ========================================================

    st.header(
        "🧠 Decision Context"
    )


    st.caption(
        "These observations describe environmental "
        "conditions used by the AgriSense decision engine. "
        "They are not independent causal explanations "
        "of the machine-learning prediction."
    )


    context1, context2, context3 = st.columns(3)


    with context1:

        st.subheader("🌧️ Rainfall")

        st.write(
            f"Risk level: "
            f"**{rainfall_risk}**"
        )

        st.write(
            get_risk_description(
                rainfall_risk
            )
        )


    with context2:

        st.subheader("🌱 Soil")

        st.write(
            f"Risk level: "
            f"**{soil_risk}**"
        )

        st.write(
            get_risk_description(
                soil_risk
            )
        )


    with context3:

        st.subheader("🛰️ Vegetation")

        st.write(
            f"Risk level: "
            f"**{vegetation_risk}**"
        )

        st.write(
            get_risk_description(
                vegetation_risk
            )
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.header(
        "💡 Recommendations"
    )


    recommendations = clean_recommendations(
        result.get(
            "recommendations",
            []
        )
    )


    if recommendations:

        for recommendation in recommendations:

            st.success(
                f"✅ {recommendation}"
            )

    else:

        st.warning(
            "No recommendations were returned "
            "by the decision engine."
        )


    # ========================================================
    # HISTORICAL CONTEXT
    # ========================================================

    st.header(
        "📊 Historical District Context"
    )


    map_data = load_map_data()


    historical_yield = None


    if not map_data.empty:

        matches = map_data[
            map_data["district"].astype(str).str.lower()
            ==
            str(result_location).lower()
        ]


        if not matches.empty:

            value = matches[
                "historical_yield"
            ].iloc[0]

            if pd.notna(value):

                historical_yield = float(
                    value
                )


    if historical_yield is not None:

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


        st.caption(
            "The historical reference is the mean yield "
            "across available records for this district."
        )


    else:

        st.info(
            "Historical yield data is not available "
            "for this district."
        )


    # ========================================================
    # ZAMBIA MAP
    # ========================================================

    st.header(
        "🗺️ Zambia Agricultural Map"
    )


    if not map_data.empty:

        map_display = map_data.copy()


        # ----------------------------------------------------
        # Point colors
        # ----------------------------------------------------

        def get_color(value):

            if pd.isna(value):

                return [
                    150,
                    150,
                    150,
                    180
                ]

            if value < 0.5:

                return [
                    220,
                    53,
                    69,
                    190
                ]

            if value < 1.0:

                return [
                    255,
                    193,
                    7,
                    190
                ]

            if value < 1.5:

                return [
                    46,
                    125,
                    91,
                    190
                ]

            return [
                25,
                118,
                210,
                190
            ]


        map_display["color"] = (
            map_display[
                "historical_yield"
            ]
            .apply(get_color)
        )


        # ----------------------------------------------------
        # Highlight selected district
        # ----------------------------------------------------

        selected_mask = (
            map_display[
                "district"
            ]
            .astype(str)
            .str.lower()
            ==
            str(result_location).lower()
        )


        map_display.loc[
            selected_mask,
            "color"
        ] = map_display.loc[
            selected_mask
        ].apply(
            lambda row:
                [
                    255,
                    87,
                    34,
                    255
                ],
            axis=1
        )


        # ----------------------------------------------------
        # MAP LAYER
        # ----------------------------------------------------

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

            "html":
                "<b>{district}</b><br/>"
                "Province: {province}<br/>"
                "Historical yield: "
                "{historical_yield} t/ha",

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

            tooltip=tooltip
        )


        st.pydeck_chart(
            deck,
            use_container_width=True
        )


        st.caption(
            "District points represent historical mean "
            "yield from the AgriSense training dataset. "
            "The selected district is highlighted."
        )


    else:

        st.info(
            "Zambia map data is currently unavailable."
        )


    # ========================================================
    # FARM SUMMARY
    # ========================================================

    st.header(
        "📋 Farm Assessment Summary"
    )


    summary = pd.DataFrame(
        [
            {
                "Crop":
                    result_crop,

                "District":
                    result_location,

                "Farm size (ha)":
                    farm_size,

                "Planting date":
                    planting_date.isoformat(),

                "Predicted yield (t/ha)":
                    round(
                        predicted_yield,
                        3
                    ),

                "Estimated production (tonnes)":
                    round(
                        total_yield,
                        3
                    ),

                "Overall risk":
                    overall_risk
            }
        ]
    )


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    csv_data = summary.to_csv(
        index=False
    )


    st.download_button(
        "📥 Download Farm Assessment",
        data=csv_data,
        file_name="agrisense_farm_assessment.csv",
        mime="text/csv"
    )


# ============================================================
# ABOUT
# ============================================================

st.divider()

st.header(
    "🌾 About AgriSense AI"
)

st.write(
    "AgriSense AI is an agricultural decision-support "
    "system designed for Zambia."
)

st.write(
    "The system combines agricultural survey data, "
    "rainfall information, soil characteristics, "
    "satellite-derived vegetation indicators and "
    "machine learning to estimate crop yield and "
    "identify environmental risk factors."
)

st.subheader(
    "Farmer-facing inputs"
)

st.write(
    "The farmer only needs:"
)

st.write(
    "- 🌾 Crop"
)

st.write(
    "- 📐 Farm size"
)

st.write(
    "- 📍 District"
)

st.write(
    "- 📅 Planting date"
)

st.write(
    "The system handles environmental feature "
    "engineering behind the scenes."
)

st.subheader(
    "Important"
)

st.warning(
    "AgriSense provides decision-support information, "
    "not a guarantee of future agricultural production. "
    "Actual yields depend on weather, management, pests, "
    "disease, seed quality and other factors."
)
