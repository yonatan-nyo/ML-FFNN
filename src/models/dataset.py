from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent

class Dataset:
    def __init__(self, filename = base_dir / '../data/datasetml_2026.csv'):
        # file is on ../dataset/datasetml_2026.csv
        self.filename = filename
        self.data = self._read_dataset()

        # validate that the data has required columns
        # cgpa,backlogs,college_tier,country,university_ranking_band,
        # internship_count,aptitude_score,
        # communication_score,specialization,industry,
        # internship_quality_score,placement_status

        required_columns = ['cgpa', 'backlogs', 'college_tier', 'country', 'university_ranking_band',
                            'internship_count', 'aptitude_score',
                            'communication_score', 'specialization', 'industry',
                            'internship_quality_score', 'placement_status']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

        # read dataset from file csv
    def _read_dataset(self):
        try:
            self.data = pd.read_csv(self.filename)
            return self.data
        except Exception as e:
            print(f"Error reading dataset: {e}")
            return None
