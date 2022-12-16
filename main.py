
from glob2 import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json


directory_path = r"/run/user/26623/gvfs/smb-share:server=physics.ox.ac.uk,share=dfs/DAQ/CondensedMatterGroups/AKGroup/Jagadish/Traces for ML"


complimentary_files_path = os.path.join(directory_path, "Complementary Traces")
noncomplimentary_files_path = os.path.join(directory_path, "Non_Complementary Traces")


complimentary_files = glob(complimentary_files_path + "*/*_gapseqML.txt")
noncomplimentary_files = glob(noncomplimentary_files_path + "*/*_gapseqML.txt")


file_path = noncomplimentary_files[1]


with open(file_path) as f:
    d = json.load(f)