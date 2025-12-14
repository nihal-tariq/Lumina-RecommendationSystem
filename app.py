import pandas as pd
from fastapi import FastAPI, Form
from typing import List


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Normalize text fields
    df["Location"] = df["Location"].astype(str).str.lower().str.strip()
    df["Sector"] = df["Sector"].astype(str).str.lower().str.strip()

    # Normalize Scholarship (robust handling)
    df["Scholarship"] = df["Scholarship"].apply(
        lambda x: 1 if str(x).strip().lower() in ["yes", "true", "1"] else 0
    )

    return df


def filter_by_subjects(df: pd.DataFrame, subjects: List[str]) -> pd.DataFrame:
    for subject in subjects:
        if subject not in df.columns:
            return pd.DataFrame()
        df = df[df[subject] == 1]
    return df


def rank_universities(
    df: pd.DataFrame,
    city: str,
    max_fee: int,
    scholarship_required: bool,
    sector: str
) -> pd.DataFrame:

    df = df.copy()
    df["priority"] = 0

    df.loc[df["Location"] == city, "priority"] += 4
    df.loc[df["Fee Structure"] <= max_fee, "priority"] += 3

    if scholarship_required:
        df.loc[df["Scholarship"] == 1, "priority"] += 2

    df.loc[df["Sector"] == sector, "priority"] += 1

    return df.sort_values(by="priority", ascending=False)


def recommend_universities(
    df: pd.DataFrame,
    subjects: List[str],
    city: str,
    max_fee: int,
    scholarship_required: bool,
    sector: str
) -> pd.DataFrame:

    # 1️⃣ Strict subject filtering
    df = filter_by_subjects(df, subjects)

    if df.empty:
        return df


    df = rank_universities(
        df,
        city,
        max_fee,
        scholarship_required,
        sector
    )


    return df[["University", "Location"]].head(4)


# ---------------------------
# FASTAPI APP
# ---------------------------
app = FastAPI(title="University Recommendation System")

# Load dataset ONCE at startup
df = load_dataset("LuminaUniversities.xlsx")


# ---------------------------
# FORM-BASED ENDPOINT
# ---------------------------
@app.post("/recommend")
def recommend(
    city: str = Form(...),
    max_fee: int = Form(...),
    subjects: str = Form(...),   # comma-separated subjects
    scholarship_required: bool = Form(...),
    sector: str = Form(...)
):
    city = city.lower()
    sector = sector.lower()
    subject_list = [s.strip() for s in subjects.split(",")]

    results = recommend_universities(
        df=df,
        subjects=subject_list,
        city=city,
        max_fee=max_fee,
        scholarship_required=scholarship_required,
        sector=sector
    )

    return {
        "recommended_universities": results.to_dict(orient="records")
    }
