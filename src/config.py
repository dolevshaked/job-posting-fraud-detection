from pathlib import Path

# rand
RANDOM_STATE = 42

# paths 
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
TEST_LABELS_PATH = RAW_DATA_DIR / "test_labels.csv"


#  cols 
TEXT_COLUMNS = [
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

CATEGORICAL_TEXT_COLUMNS = [
    "title",
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "location"
]

TEXT_PREPROCESS_COLUMNS = TEXT_COLUMNS + CATEGORICAL_TEXT_COLUMNS

# cols to drop 
DROP_COLUMNS = ["salary_range", "department", "job_id"]

# cols to fill nulls 
CATEGORICAL_NULL_COLUMNS = [
    "location",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

TEXT_NULL_COLUMNS = [
    "benefits",
    "requirements",
    "description",
    "company_profile",
]

# high simensionality categorical cols to cluster 
CLUSTER_COLUMNS = ["title", "location", "industry", "function"]

# ordinal encoding 
ORDINAL_CATEGORY_ORDERS = {
    "employment_type": ["temporary", "contract", "part-time", "full-time"],
    "required_experience": ['applicable', 'internship', 'entry level', 'associate', 'mid-senior level', 'director', 'executive'],
}

EDUCATION_MAP = {
    'unspecified': 0,
    'high school equivalent': 1,
    'high school coursework': 1,
    'certification': 2,
    'vocational - h diploma': 2,
    'vocational': 2,
    'vocational - degree': 2,
    'college coursework completed': 2,
    'associate degree': 3,
    "bachelor 's degree": 4,
    'professional': 5,
    "master 's degree": 6,
    'doctorate': 7
}

