#%%
import pandas as pd

df = pd.read_json("./benchmark_results/phi35_new_prompt_results_20250214_045248.jsonl", lines=True)

#%%
df.head()

#%%
random_index = 4
print(df['actual_category'].iloc[random_index])
print(df['model_response'].iloc[random_index])
