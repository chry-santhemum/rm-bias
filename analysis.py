# %%

import pickle
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm
from state import SeedState

# %%
hyperparam_choices = [
    "b10-n6",
    "b6-n10",
    "b3-n20",
    "N10-M2-K5-n2",
    "N4-M5-K5-n2",
    "N4-M2-K3-n5",
    "N3-M2-K2-n10",
]

# look for directory name
all_data = {}
for suffix in tqdm(hyperparam_choices, desc="Loading data"):
    for dir_name in ["bon_iter", "evo"]:
        try:
            dir_path = next(Path(f"data/{dir_name}").glob(f"*{suffix}"))
            file_path = next(dir_path.glob(f"step_*.pkl"))

            with open(file_path, "rb") as f:
                seed_states = pickle.load(f)
            
            all_data[suffix] = seed_states
            break
        
        except StopIteration:
            print(f"File not found for {suffix} in {dir_name}...")
            continue

# %%
def get_max_score(seed_state: SeedState):
    step_stats = []
    for step_idx, step_content in enumerate(seed_state.history):
        step_stats.extend([
            (system_prompt, stats.mean_score, stats.stdev_score)
            for system_prompt, stats in step_content.items()
        ])

    step_stats.sort(key=lambda x: x[1], reverse=True)
    return step_stats[0][1]

all_max_scores = {}
for suffix, seed_states in all_data.items():
    all_max_scores[suffix] = [get_max_score(seed_state) for seed_state in seed_states]

all_max_scores["indices"] = [str(seed_state.index) for seed_state in seed_states]

df = pd.DataFrame(all_max_scores)  # type: ignore

# %%

color_codes = pd.factorize(df['indices'])[0]
colors = px.colors.qualitative.G10
custom_colorscale = []
for i, color in enumerate(colors):
    custom_colorscale.append(color)

fig = go.Figure(data=
    go.Parcoords(
        line=dict(
            color=color_codes,
            colorscale=custom_colorscale,
            showscale=False,
        ),
        dimensions=[dict(label=col, values=df[col]) for col in hyperparam_choices],
    )
)

fig.update_layout(title_text="Sweep")
fig.show()

# %%
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

plt.figure(figsize=(10, 6))
parallel_coordinates(
    df,
    'indices',
    colormap='viridis',
    linewidth=4  # <-- Set the line width here
)
plt.title('Sweep')
plt.ylabel('Adversarial Score')
plt.xlabel('Hyperparameter')
plt.xticks(rotation=45)
plt.legend(title='Seeds', loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.show()


# %%
def get_best_score(seed_state) -> float:
    step_max = []

    for step_idx, step_content in enumerate(seed_state.history):
        step_stats = [
            (system_prompt, stats.mean_score, stats.stdev_score)
            for system_prompt, stats in step_content.items()
        ]

        step_stats.sort(key=lambda x: x[1], reverse=True)
        step_max.append(step_stats[0][1])
    
    return max(step_max)
# %%
