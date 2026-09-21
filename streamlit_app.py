import io
from datetime import date

import pandas as pd
import requests
import streamlit as st
import pydeck as pdk

# PDF reporting
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True

except ImportError:
    REPORTLAB_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://agrisense-fyvu.onrender.com/predict"

st.set_page_config(
    page_title="AgriSense AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MAP COORDINATES
# ============================================================

# These are geographic reference points for map context.
# If a district is not listed, the app deliberately falls back
# to a Zambia-level map instead of inventing coordinates.

DISTRICT_COORDS = {
    "Chibombo": (-14.857, 27.640),
    "Chisamba": (-14.772, 28.459),
    "Chitambo": (-12.543, 30.400),
    "Kabwe": (-14.458, 28.400),
    "Kapiri Mposhi": (-14.196, 28.487),
    "Lusaka": (-15.4167, 28.2833),
    "Ndola": (-12.9667, 28.6333),
    "Kitwe": (-12.8167, 28.2000),
    "Chingola": (-12.52897, 27.88382),
    "Mufulira": (-12.5498, 28.2407),
    "Livingstone": (-17.8419, 25.8543),
    "Chipata": (-13.6333, 32.6500),
    "Chadiza": (-14.0667, 32.4333),
    "Petauke": (-14.2500, 31.3333),
    "Katete": (-14.0667, 31.9500),
    "Mongu": (-15.2740, 23.1370),
    "Kaoma": (-14.8000, 24.8000),
    "Sesheke": (-17.4759, 24.2968),
    "Mazabuka": (-15.8560, 27.7480),
    "Monze": (-16.2833, 27.4833),
    "Choma": (-16.8089, 26.9875),
    "Kalomo": (-17.0333, 26.4833),
    "Kasama": (-10.2129, 31.1808),
    "Mbala": (-8.8400, 31.3650),
    "Mansa": (-11.1998, 28.8943),
    "Samfya": (-11.3649, 29.5565),
    "Kawambwa": (-9.7917, 29.0794),
    "Solwezi": (-12.1688, 26.3894),
    "Mwinilunga": (-11.7358, 24.4293),
    "Mkushi": (-13.6200, 29.3939),
    "Serenje": (-13.2325, 30.2339),
}


# ============================================================
# CROP OPTIONS
# ============================================================

CROP_OPTIONS = [
    "Maize(for grain)",
    "Groundnuts(unshelled)",
    "Sunflower (for grain)",
    "Soya beans (for grain)",
    "Sweet Potatoes (for tuber)",
    "Mixed Beans (for grain)",
    "Finger Millet",
    "Cow Peas (for grain)",
    "Rice",
    "Sorghum(for grain)",
    "Bambara nuts",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_recommendations(value):
    """
    Converts whatever the API returns into a clean list.
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


def safe_float(value, default=None):
    """
    Safely convert a value to float.
    """
    try:
        if value is None or value == "":
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def first_value(data, keys, default=None):
    """
    Return the first available value from a dictionary.
    """
    if not isinstance(data, dict):
        return default

    for key in keys:
        if key in data:

            value = data[key]

            if value is not None and value != "":
                return value

    return default


def normalize_risk(value):
    """
    Make risk labels look consistent.
    """
    if value is None:
        return "Not available"

    text = str(value).strip()

    if not text:
        return "Not available"

    return text.title()


def risk_message(risk):
    """
    Simple explanation for users.
    """
    value = str(risk).lower()

    if "high" in value:
        return (
            "This indicator is in the higher-risk range "
            "defined by the current assessment rules."
        )

    if "moderate" in value or "medium" in value:
        return (
            "This indicator is in a moderate range "
            "and deserves monitoring."
        )

    if "low" in value:
        return (
            "This indicator is not currently in the "
            "higher-risk range defined by the assessment rules."
        )

    return "No risk interpretation was returned."


def flatten_environment(environment):
    """
    Convert environmental_features into a simple table.
    Handles both simple values and nested dictionaries.
    """

    rows = []

    if not isinstance(environment, dict):
        return rows

    for key, value in environment.items():

        label = (
            str(key)
            .replace("_", " ")
            .strip()
            .title()
        )

        if isinstance(value, dict):

            actual = first_value(
                value,
                [
                    "value",
                    "mean",
                    "current",
                    "estimate",
                    "amount",
                ],
                None,
            )

            status = first_value(
                value,
                [
                    "status",
                    "risk",
                    "condition",
                    "level",
                ],
                None,
            )

            if actual is None:
                actual = str(value)

            display = str(actual)

            if status:
                display = (
                    f"{display} "
                    f"({normalize_risk(status)})"
                )

            rows.append(
                {
                    "Indicator": label,
                    "Value": display,
                }
            )

        else:

            rows.append(
                {
                    "Indicator": label,
                    "Value": str(value),
                }
            )

    return rows


def build_map(district):
    """
    Build a simple geographic context map.
    """

    coords = DISTRICT_COORDS.get(
        district.strip()
    )

    # District coordinate available
    if coords:

        lat, lon = coords

        view_state = pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=7.5,
            pitch=0,
        )

        data = pd.DataFrame(
            [
                {
                    "district": district,
                    "latitude": lat,
                    "longitude": lon,
                }
            ]
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=data,
            get_position="[longitude, latitude]",
            get_radius=18000,
            pickable=True,
        )

        return pdk.Deck(
            map_style=None,
            initial_view_state=view_state,
            layers=[layer],
            tooltip={
                "text": "{district}"
            },
        )

    # Zambia-level fallback
    view_state = pdk.ViewState(
        latitude=-13.1339,
        longitude=27.8493,
        zoom=4.6,
        pitch=0,
    )

    return pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=[],
    )


# ============================================================
# PDF REPORT
# ============================================================

def build_pdf(
    result,
    crop,
    farm_size,
    district,
    planting_date,
):

    if not REPORTLAB_AVAILABLE:
        return None

    predicted_yield = safe_float(
        first_value(
            result,
            [
                "predicted_yield_tonnes_per_ha",
                "predicted_yield",
                "yield",
            ],
        ),
        0.0,
    )

    estimated_total = safe_float(
        first_value(
            result,
            [
                "estimated_total_yield_tonnes",
                "estimated_production_tonnes",
            ],
        ),
        None,
    )

    overall_risk = normalize_risk(
        first_value(
            result,
            [
                "risk_level",
                "overall_risk",
            ],
            "Not available",
        )
    )

    rainfall_risk = normalize_risk(
        first_value(
            result,
            ["rainfall_risk"],
            "Not available",
        )
    )

    soil_risk = normalize_risk(
        first_value(
            result,
            ["soil_risk"],
            "Not available",
        )
    )

    vegetation_risk = normalize_risk(
        first_value(
            result,
            ["vegetation_risk"],
            "Not available",
        )
    )

    recommendations = clean_recommendations(
        result.get("recommendations")
    )

    environment_rows = flatten_environment(
        result.get(
            "environmental_features",
            {},
        )
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=12,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
    )

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    story = []

    story.append(
        Paragraph(
            "🌾 AgriSense AI",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Farm Assessment Report",
            ParagraphStyle(
                "Subtitle",
                parent=styles["Heading2"],
                alignment=TA_CENTER,
            ),
        )
    )

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------------------------------
    # FARM DETAILS
    # --------------------------------------------------------

    farmer_table = Table(
        [
            ["Crop", str(crop)],
            [
                "Farm size",
                f"{farm_size:.2f} ha",
            ],
            ["District", str(district)],
            [
                "Planting date",
                str(planting_date),
            ],
        ],
        colWidths=[130, 350],
    )

    farmer_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        Paragraph(
            "Farm Information",
            styles["Heading2"],
        )
    )

    story.append(farmer_table)

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction_table = Table(
        [
            [
                "Predicted yield",
                (
                    f"{predicted_yield:.3f} tonnes/ha"
                    if predicted_yield is not None
                    else "Not returned"
                ),
            ],
            [
                "Estimated production",
                (
                    f"{estimated_total:.3f} tonnes"
                    if estimated_total is not None
                    else "Not returned"
                ),
            ],
            [
                "Overall risk",
                overall_risk,
            ],
        ],
        colWidths=[180, 300],
    )

    prediction_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        Paragraph(
            "Assessment",
            styles["Heading2"],
        )
    )

    story.append(
        prediction_table
    )

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    risk_table = Table(
        [
            [
                "Indicator",
                "Assessment",
            ],
            [
                "Rainfall",
                rainfall_risk,
            ],
            [
                "Soil",
                soil_risk,
            ],
            [
                "Vegetation",
                vegetation_risk,
            ],
        ],
        colWidths=[180, 300],
    )

    risk_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        Paragraph(
            "Environmental Risk",
            styles["Heading2"],
        )
    )

    story.append(
        risk_table
    )

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Recommendations",
            styles["Heading2"],
        )
    )

    if recommendations:

        for recommendation in recommendations:

            story.append(
                Paragraph(
                    f"• {recommendation}",
                    styles["BodyText"],
                )
            )

            story.append(
                Spacer(1, 4)
            )

    else:

        story.append(
            Paragraph(
                "No recommendations were returned.",
                styles["BodyText"],
            )
        )

    # --------------------------------------------------------
    # ENVIRONMENTAL FEATURES
    # --------------------------------------------------------

    if environment_rows:

        story.append(
            Spacer(1, 12)
        )

        story.append(
            Paragraph(
                "Environmental Features",
                styles["Heading2"],
            )
        )

        table_data = [
            ["Indicator", "Value"]
        ]

        for row in environment_rows:

            table_data.append(
                [
                    row["Indicator"],
                    row["Value"],
                ]
            )

        environment_table = Table(
            table_data,
            colWidths=[250, 230],
        )

        environment_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        story.append(
            environment_table
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    story.append(
        Spacer(1, 18)
    )

    story.append(
        Paragraph(
            "<b>Important:</b> AgriSense AI provides "
            "model-based decision support. The assessment "
            "is an estimate and should not be treated as "
            "a guaranteed farm outcome or a replacement "
            "for local agricultural expertise.",
            small_style,
        )
    )

    story.append(
        Spacer(1, 8)
    )

    story.append(
        Paragraph(
            "Project data sources include ZamStats, CHIRPS, "
            "SoilGrids and MODIS-derived vegetation indicators.",
            small_style,
        )
    )

    document.build(story)

    return buffer.getvalue()


# ============================================================
# HEADER
# ============================================================

st.title("🌾 AgriSense AI")

st.subheader(
    "Intelligent Agricultural Risk, Yield Prediction "
    "& Decision Support"
)

st.caption(
    "A Zambia-focused decision-support system combining "
    "agricultural, rainfall, soil and vegetation information."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌱 Farm Information")

    crop = st.selectbox(
        "Crop",
        CROP_OPTIONS,
    )

    farm_size = st.number_input(
        "Farm size (hectares)",
        min_value=0.01,
        value=2.50,
        step=0.10,
    )

    district = st.text_input(
        "District",
        value="Chibombo",
        help="Enter the Zambia district for the farm.",
    ).strip()

    planting_date = st.date_input(
        "Planting date",
        value=date(
            2026,
            11,
            15,
        ),
    )

    assess = st.button(
        "🔍 Assess Farm",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.caption(
        "AgriSense AI is a model-based agricultural "
        "decision-support tool."
    )

    st.caption(
        "Environmental information should be interpreted "
        "as available historical/model inputs rather than "
        "guaranteed real-time observations."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "assessment" not in st.session_state:

    st.session_state.assessment = None


# ============================================================
# CALL API
# ============================================================

if assess:

    if not district:

        st.error(
            "Please enter a district."
        )

        st.stop()

    payload = {

        "crop": crop,

        "farm_size_ha": float(
            farm_size
        ),

        "location": district,

        "planting_date": planting_date.isoformat(),
    }

    with st.spinner(
        "Running the AgriSense assessment..."
    ):

        try:

            response = requests.post(
                API_URL,
                json=payload,
                timeout=90,
            )

            if response.status_code == 200:

                st.session_state.assessment = (
                    response.json()
                )

                st.session_state.assessment_input = (
                    payload
                )

            else:

                try:
                    detail = response.json()

                except ValueError:
                    detail = response.text

                st.session_state.assessment = None

                st.error(
                    f"The API returned HTTP "
                    f"{response.status_code}.\n\n"
                    f"Response: {detail}"
                )

        except requests.exceptions.Timeout:

            st.session_state.assessment = None

            st.error(
                "The API took too long to respond. "
                "The cloud service may be waking up. "
                "Please try again."
            )

        except requests.exceptions.RequestException as exc:

            st.session_state.assessment = None

            st.error(
                f"Could not reach the AgriSense API: {exc}"
            )


# ============================================================
# GET RESULT
# ============================================================

result = st.session_state.assessment


# ============================================================
# EMPTY STATE
# ============================================================

if result is None:

    st.info(
        "Enter the farm information in the sidebar "
        "and click **Assess Farm** to generate an assessment."
    )

    st.header(
        "How AgriSense AI works"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "1",
            "Farmer input",
        )

        st.caption(
            "Crop, farm size, district and planting date."
        )

    with c2:

        st.metric(
            "2",
            "Environmental data",
        )

        st.caption(
            "Rainfall, soil and vegetation indicators."
        )

    with c3:

        st.metric(
            "3",
            "ML prediction",
        )

        st.caption(
            "XGBoost estimates expected yield."
        )

    with c4:

        st.metric(
            "4",
            "Decision support",
        )

        st.caption(
            "Risk information and recommendations."
        )

    st.divider()

    st.header(
        "📡 Data Sources"
    )

    source_df = pd.DataFrame(
        [
            [
                "ZamStats",
                "Historical agricultural production and yield",
            ],
            [
                "CHIRPS",
                "Historical rainfall information",
            ],
            [
                "SoilGrids",
                "Soil properties",
            ],
            [
                "MODIS NDVI",
                "Satellite-derived vegetation indicators",
            ],
        ],
        columns=[
            "Source",
            "What it provides",
        ],
    )

    st.dataframe(
        source_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "AgriSense combines these sources into model "
        "features rather than relying on a single dataset."
    )

    st.stop()


# ============================================================
# INPUT DATA
# ============================================================

input_data = st.session_state.get(
    "assessment_input",
    {
        "crop": crop,
        "farm_size_ha": farm_size,
        "location": district,
        "planting_date": planting_date.isoformat(),
    },
)


# ============================================================
# RESULT VALUES
# ============================================================

result_crop = result.get(
    "crop",
    input_data["crop"],
)

result_location = result.get(
    "location",
    input_data["location"],
)

predicted_yield = safe_float(
    first_value(
        result,
        [
            "predicted_yield_tonnes_per_ha",
            "predicted_yield",
            "yield",
        ],
    ),
    None,
)

estimated_total = safe_float(
    first_value(
        result,
        [
            "estimated_total_yield_tonnes",
            "estimated_production_tonnes",
        ],
    ),
    None,
)

# Safety fallback:
# If API doesn't return total production,
# calculate it from predicted yield × farm size.

if (
    estimated_total is None
    and predicted_yield is not None
):

    estimated_total = (
        predicted_yield
        * float(
            input_data["farm_size_ha"]
        )
    )


overall_risk = normalize_risk(
    first_value(
        result,
        [
            "risk_level",
            "overall_risk",
        ],
        "Not available",
    )
)

rainfall_risk = normalize_risk(
    first_value(
        result,
        ["rainfall_risk"],
        "Not available",
    )
)

soil_risk = normalize_risk(
    first_value(
        result,
        ["soil_risk"],
        "Not available",
    )
)

vegetation_risk = normalize_risk(
    first_value(
        result,
        ["vegetation_risk"],
        "Not available",
    )
)

recommendations = clean_recommendations(
    result.get("recommendations")
)

environmental_features = result.get(
    "environmental_features",
    {},
)


# ============================================================
# FARM SUMMARY
# ============================================================

st.header(
    "🌾 Farm Assessment"
)

st.write(
    f"**{result_crop}** • "
    f"**{result_location}** • "
    f"**{float(input_data['farm_size_ha']):.2f} ha** • "
    f"Planting date: **{input_data['planting_date']}**"
)


# ============================================================
# MAIN METRICS
# ============================================================

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Predicted yield",
        (
            f"{predicted_yield:.3f} t/ha"
            if predicted_yield is not None
            else "N/A"
        ),
    )

with m2:

    st.metric(
        "Estimated production",
        (
            f"{estimated_total:.3f} t"
            if estimated_total is not None
            else "N/A"
        ),
    )

with m3:

    st.metric(
        "Overall risk",
        overall_risk,
    )


# ============================================================
# RISK BREAKDOWN
# ============================================================

st.divider()

st.header(
    "🌦️ Environmental Risk"
)

r1, r2, r3 = st.columns(3)

with r1:

    st.subheader(
        "🌧 Rainfall"
    )

    st.metric(
        "Risk",
        rainfall_risk,
    )

    st.caption(
        risk_message(
            rainfall_risk
        )
    )


with r2:

    st.subheader(
        "🌱 Soil"
    )

    st.metric(
        "Risk",
        soil_risk,
    )

    st.caption(
        risk_message(
            soil_risk
        )
    )


with r3:

    st.subheader(
        "🛰 Vegetation"
    )

    st.metric(
        "Risk",
        vegetation_risk,
    )

    st.caption(
        risk_message(
            vegetation_risk
        )
    )


# ============================================================
# WHAT DOES THIS MEAN?
# ============================================================

st.divider()

st.header(
    "💡 What does this mean?"
)

if (
    predicted_yield is not None
    and estimated_total is not None
):

    st.info(
        f"The model estimates approximately "
        f"**{predicted_yield:.3f} tonnes per hectare**, "
        f"which corresponds to about "
        f"**{estimated_total:.3f} tonnes** "
        f"for the entered farm size."
    )

else:

    st.info(
        "The API returned an assessment, but a complete "
        "numerical yield summary was not available."
    )

st.caption(
    "This is a model-based estimate. "
    "It is not a guarantee of the actual harvest."
)


# ============================================================
# RECOMMENDATIONS
# ============================================================

st.header(
    "🧭 Recommendations"
)

if recommendations:

    for recommendation in recommendations:

        st.success(
            f"✅ {recommendation}"
        )

else:

    st.info(
        "No additional recommendations were returned."
    )


# ============================================================
# ENVIRONMENTAL FEATURES
# ============================================================

st.divider()

st.header(
    "📊 Environmental Conditions"
)

environment_rows = flatten_environment(
    environmental_features
)

if environment_rows:

    environment_df = pd.DataFrame(
        environment_rows
    )

    st.dataframe(
        environment_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "The API did not return detailed environmental "
        "feature values."
    )


# ============================================================
# HISTORICAL CONTEXT
# ============================================================

st.header(
    "📚 Historical District Context"
)

historical_mean = safe_float(
    first_value(
        result,
        [
            "historical_district_mean_yield",
            "historical_mean_yield",
            "district_historical_mean",
        ],
    ),
    None,
)

if historical_mean is not None:

    st.metric(
        "District historical yield context",
        f"{historical_mean:.3f} t/ha",
    )

    st.caption(
        "This is district-wide historical context "
        "across the available records. It should not "
        "be interpreted as a crop-specific benchmark "
        "unless crop-and-district history is explicitly "
        "returned by the API."
    )

    if predicted_yield is not None:

        difference = (
            predicted_yield
            - historical_mean
        )

        st.write(
            "Model estimate relative to district "
            f"context: **{difference:+.3f} t/ha**."
        )

else:

    st.info(
        "A district historical reference was not "
        "returned by the API. No crop-specific "
        "historical benchmark is shown."
    )


# ============================================================
# MAP
# ============================================================

st.divider()

st.header(
    "📍 Zambia Geographic Context"
)

st.pydeck_chart(
    build_map(
        result_location
    ),
    use_container_width=True,
)

if result_location in DISTRICT_COORDS:

    st.caption(
        f"Map marker shows the available geographic "
        f"reference for {result_location}. "
        "This is geographic context, not a farm boundary."
    )

else:

    st.caption(
        "A precise frontend coordinate was not available "
        "for this district, so the map remains at Zambia "
        "level instead of inventing a location."
    )


# ============================================================
# DATA AVAILABILITY
# ============================================================

st.divider()

st.header(
    "📡 Data Availability"
)

environment_text = str(
    environmental_features
).lower()

availability = {

    "Farmer inputs": True,

    "Location": bool(
        result_location
    ),

    "Rainfall": (
        "rainfall"
        in environment_text
    ),

    "Soil": (
        "soil"
        in environment_text
    ),

    "Vegetation": (
        "vegetation"
        in environment_text
        or "ndvi"
        in environment_text
    ),
}

availability_rows = []

for component, available in availability.items():

    availability_rows.append(
        [
            component,
            (
                "Available"
                if available
                else "Limited / unavailable"
            ),
        ]
    )

availability_df = pd.DataFrame(
    availability_rows,
    columns=[
        "Component",
        "Status",
    ],
)

st.dataframe(
    availability_df,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "This is a data-availability indicator, not "
    "a statistical prediction-confidence percentage. "
    "AgriSense does not claim a numerical confidence "
    "score here."
)


# ============================================================
# TECHNICAL / JUDGE VIEW
# ============================================================

st.divider()

with st.expander(
    "🔬 Technical Details / Judge View"
):

    st.subheader(
        "Machine Learning Model"
    )

    technical_df = pd.DataFrame(
        [
            [
                "Algorithm",
                "XGBoost Regression",
            ],
            [
                "Target",
                "Crop yield (tonnes/hectare)",
            ],
            [
                "Training records",
                "31,928",
            ],
            [
                "Validation",
                "District-based geographic validation",
            ],
            [
                "MAE",
                "0.7162 tonnes/hectare",
            ],
            [
                "RMSE",
                "1.0258 tonnes/hectare",
            ],
            [
                "R²",
                "0.2977",
            ],
        ],
        columns=[
            "Item",
            "Value",
        ],
    )

    st.dataframe(
        technical_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "System Architecture"
    )

    st.code(
        """Farmer Input
      ↓
Agricultural + Environmental Data
      ↓
Feature Engineering
      ↓
XGBoost Regression
      ↓
Predicted Yield
      ↓
Risk Assessment
      ↓
Recommendations
      ↓
FastAPI
      ↓
Streamlit
""",
        language="text",
    )

    st.subheader(
        "Why geographic validation?"
    )

    st.write(
        "A random split can place records from the same "
        "geographical areas into both training and testing "
        "data. We therefore also used district-based "
        "validation where the test districts were not "
        "present in the training data."
    )

    st.subheader(
        "Data Sources"
    )

    technical_sources = pd.DataFrame(
        [
            [
                "ZamStats",
                "Agricultural production and yield records",
            ],
            [
                "CHIRPS",
                "Rainfall information",
            ],
            [
                "SoilGrids",
                "Soil properties",
            ],
            [
                "MODIS NDVI",
                "Vegetation indicators",
            ],
            [
                "Zambia administrative data",
                "Geographic/district reference",
            ],
        ],
        columns=[
            "Source",
            "Purpose",
        ],
    )

    st.dataframe(
        technical_sources,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Important Limitations"
    )

    limitations = [

        "The model provides estimates rather than guaranteed harvest outcomes.",

        "Historical agricultural records do not represent every individual farm.",

        "Environmental datasets have different spatial and temporal resolutions.",

        "Some environmental observations are unavailable for some historical records.",

        "The current NDVI development data covers a limited historical period.",

        "Geographic validation performance is lower than random-split performance.",

        "The current application is not a real-time weather forecasting system.",

        "Recommendations are decision-support suggestions and should be considered together with local agricultural knowledge.",
    ]

    for limitation in limitations:

        st.write(
            f"• {limitation}"
        )


# ============================================================
# PDF FARM ASSESSMENT
# ============================================================

st.divider()

st.header(
    "📄 Farm Assessment Report"
)

if REPORTLAB_AVAILABLE:

    report_bytes = build_pdf(
        result=result,
        crop=result_crop,
        farm_size=float(
            input_data["farm_size_ha"]
        ),
        district=result_location,
        planting_date=input_data["planting_date"],
    )

    if report_bytes:

        safe_crop = "".join(
            character
            if character.isalnum()
            else "_"
            for character in str(
                result_crop
            )
        ).strip("_") or "farm"

        safe_district = "".join(
            character
            if character.isalnum()
            else "_"
            for character in str(
                result_location
            )
        ).strip("_") or "district"

        filename = (
            f"AgriSense_Assessment_"
            f"{safe_crop}_"
            f"{safe_district}.pdf"
        )

        st.download_button(
            label="📥 Download Farm Assessment PDF",
            data=report_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
        )

else:

    st.warning(
        "PDF reporting is unavailable because "
        "reportlab is not installed. Add reportlab "
        "to requirements.txt and redeploy."
    )


# ============================================================
# ABOUT
# ============================================================

st.divider()

st.header(
    "🌾 About AgriSense AI"
)

st.write(
    "AgriSense AI turns agricultural and environmental "
    "information into simple decision-support information. "
    "The system combines historical agricultural records "
    "with rainfall, soil and vegetation indicators to "
    "estimate expected yield and identify environmental "
    "conditions that deserve attention."
)

st.caption(
    "The goal is not to replace farmer knowledge. "
    "The goal is to provide additional evidence that "
    "can support agricultural planning and monitoring."
)

st.caption(
    "AgriSense AI • Zambia-focused agricultural "
    "decision support • Model-based assessment"
)
