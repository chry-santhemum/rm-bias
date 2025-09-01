# %%
import time
import pickle
import json
import random
from tqdm.auto import tqdm
from datasets import load_dataset
import asyncio
import nest_asyncio

nest_asyncio.apply()

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import community_detection
from sklearn.cluster import MiniBatchKMeans

import numpy as np

from client import get_universal_caller, sample_from_model_parallel

logging.set_verbosity(logging.INFO)


from datasets import load_dataset

ds = load_dataset("allenai/WildChat-1M")