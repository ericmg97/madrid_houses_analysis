from pathlib import Path

# data loading configuration
DATA_DIR = (Path(__file__).parent.parent.parent / 'data').resolve()
RAW_DIR = DATA_DIR / 'raw'
SUBMISSIONS_DIR = DATA_DIR / 'submissions'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = DATA_DIR / 'models'