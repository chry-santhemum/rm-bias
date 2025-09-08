"""Cluster prompts in WildChat"""


# %%
import time
import pickle
import json
import random
import logging
from tqdm.auto import tqdm
import asyncio
import nest_asyncio
nest_asyncio.apply()

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import community_detection
from sklearn.cluster import MiniBatchKMeans


from client import get_universal_caller, sample_from_model_parallel

logging.getLogger(__name__)



ds = load_dataset("allenai/WildChat-1M")