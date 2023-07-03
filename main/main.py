from evaluation.confusion import Evaluation
from modeling.classification_models import ClassificationModel
from preprocessing.dimensionality_reduction import DR
from preprocessing.read_data import ReadData, Format

# ---------- Preprocessing--------------
read_data = ReadData()
dr = DR()
df = read_data.read_data_from_local_path("MCSDatasetNEXTCONLab.csv", Format.CSV)
print(df.head(10))
# ---------- Modeling--------------
modeling = ClassificationModel(..., ..., ...)

# ---------- Visualization--------------

# ---------- Evaluation--------------
evaluation = Evaluation(..., ...)
