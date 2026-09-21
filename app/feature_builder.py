from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data/master/agrisense_master_v5.csv"


CATEGORICAL_FEATURES = [
    "crop",
    "province",
    "district",
]


NUMERIC_FEATURES = [
    "farm_size_ha",
    "planting_month",

    "rainfall_planting_month",
    "rainfall_1_month_before",
    "rainfall_2_months_before",
    "rainfall_3_months_before",
    "rainfall_preplant_3m_total",
    "rainfall_preplant_3m_mean",
    "preplant_rain_months_available",

    "soil_ph",
    "soil_organic_carbon_g_kg",
    "soil_nitrogen_g_kg",
    "soil_clay_percent",
    "soil_sand_percent",
    "soil_silt_percent",

    "ndvi_preplant_mean",
    "ndvi_preplant_max",
    "ndvi_preplant_min",
    "ndvi_preplant_std",
    "ndvi_preplant_median",
    "ndvi_preplant_first",
    "ndvi_preplant_last",
    "ndvi_preplant_observations",
    "ndvi_preplant_change",
]


MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


class FeatureBuilder:

    def __init__(self):
        self.df = pd.read_csv(DATA_PATH)

        self.df["planting_date"] = pd.to_datetime(
            self.df["planting_date"],
            errors="coerce"
        )

        self.df["planting_month"] = (
            self.df["planting_date"].dt.month
        )

        self._clean_strings()

        self.district_province = (
            self.df
            .groupby("district")["province"]
            .agg(lambda x: x.mode().iloc[0])
            .to_dict()
        )

        self.crop_values = sorted(
            self.df["crop"].dropna().unique().tolist()
        )

        self.district_values = sorted(
            self.df["district"].dropna().unique().tolist()
        )

    def _clean_strings(self):

        for col in [
            "crop",
            "province",
            "district"
        ]:
            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.strip()
            )

    def resolve_district(self, location):

        location = location.strip()

        # Exact match
        exact = [
            d for d in self.district_values
            if d.lower() == location.lower()
        ]

        if exact:
            return exact[0]

        # Partial match
        partial = [
            d for d in self.district_values
            if location.lower() in d.lower()
        ]

        if len(partial) == 1:
            return partial[0]

        if len(partial) > 1:
            raise ValueError(
                f"Ambiguous location '{location}'. "
                f"Possible matches: {partial[:10]}"
            )

        raise ValueError(
            f"District '{location}' was not found."
        )

    def resolve_crop(self, crop):

        crop = crop.strip()

        exact = [
            c for c in self.crop_values
            if c.lower() == crop.lower()
        ]

        if exact:
            return exact[0]

        partial = [
            c for c in self.crop_values
            if crop.lower() in c.lower()
        ]

        if len(partial) == 1:
            return partial[0]

        if len(partial) > 1:
            raise ValueError(
                f"Ambiguous crop '{crop}'. "
                f"Possible matches: {partial[:10]}"
            )

        raise ValueError(
            f"Crop '{crop}' was not found."
        )

    def _profile(self, frame):

        if len(frame) == 0:
            return None

        profile = {}

        for column in NUMERIC_FEATURES:

            if column == "farm_size_ha":
                continue

            if column == "planting_month":
                continue

            if column not in frame.columns:
                continue

            values = pd.to_numeric(
                frame[column],
                errors="coerce"
            ).dropna()

            if len(values) > 0:
                profile[column] = float(values.median())

        return profile

    def _find_profile(
        self,
        crop,
        district,
        planting_month
    ):

        df = self.df

        # --------------------------------------------------
        # LEVEL 1
        # Crop + District + Month
        # --------------------------------------------------

        subset = df[
            (df["crop"] == crop)
            & (df["district"] == district)
            & (df["planting_month"] == planting_month)
        ]

        profile = self._profile(subset)

        if profile:
            return profile, "crop_district_month"

        # --------------------------------------------------
        # LEVEL 2
        # Crop + District
        # --------------------------------------------------

        subset = df[
            (df["crop"] == crop)
            & (df["district"] == district)
        ]

        profile = self._profile(subset)

        if profile:
            return profile, "crop_district"

        # --------------------------------------------------
        # LEVEL 3
        # District + Month
        # --------------------------------------------------

        subset = df[
            (df["district"] == district)
            & (df["planting_month"] == planting_month)
        ]

        profile = self._profile(subset)

        if profile:
            return profile, "district_month"

        # --------------------------------------------------
        # LEVEL 4
        # District
        # --------------------------------------------------

        subset = df[
            df["district"] == district
        ]

        profile = self._profile(subset)

        if profile:
            return profile, "district"

        # --------------------------------------------------
        # LEVEL 5
        # Crop
        # --------------------------------------------------

        subset = df[
            df["crop"] == crop
        ]

        profile = self._profile(subset)

        if profile:
            return profile, "crop"

        # --------------------------------------------------
        # LEVEL 6
        # Global
        # --------------------------------------------------

        profile = self._profile(df)

        return profile, "global"

    def build_features(
        self,
        crop,
        farm_size_ha,
        location,
        planting_date
    ):

        crop = self.resolve_crop(crop)

        district = self.resolve_district(location)

        planting_date = pd.to_datetime(
            planting_date,
            errors="raise"
        )

        planting_month = int(
            planting_date.month
        )

        province = self.district_province.get(
            district
        )

        if province is None:
            raise ValueError(
                f"Province could not be resolved "
                f"for district '{district}'."
            )

        profile, profile_level = self._find_profile(
            crop=crop,
            district=district,
            planting_month=planting_month
        )

        row = {
            "crop": crop,
            "province": province,
            "district": district,

            "farm_size_ha": float(
                farm_size_ha
            ),

            "planting_month": planting_month,
        }

        for column in NUMERIC_FEATURES:

            if column in [
                "farm_size_ha",
                "planting_month"
            ]:
                continue

            row[column] = profile.get(
                column,
                None
            )

        features = pd.DataFrame(
            [row],
            columns=MODEL_FEATURES
        )

        metadata = {
            "crop": crop,
            "district": district,
            "province": province,
            "planting_month": planting_month,
            "profile_source": profile_level,
        }

        return features, metadata


feature_builder = FeatureBuilder()
