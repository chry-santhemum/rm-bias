# %%
if __name__ == "__main__":
    import pandas as pd
    import hashlib
    import random
    import json
    from datasets import load_dataset
    from tqdm.auto import tqdm
    from pathlib import Path
    from standard_prompts import set_seed_all

    set_seed_all(10086)

    # hf_instruction_test = load_dataset("HuggingFaceH4/instruction-dataset", split="test")
    # prompts_to_sample = list(hf_instruction_test["prompt"])



    cluster_df: pd.DataFrame = pd.read_csv("data/wildchat/cluster_50k.csv")
    labels_df: pd.DataFrame = pd.read_csv("data/wildchat/labels_50k.csv")
    # prompts_to_sample = []

    for topic_id in tqdm(range(1, 30), desc="Processing topics"):
        topic = cluster_df.loc[cluster_df.index[topic_id+1], "Name"].split('_', maxsplit=1)[-1]  # description
        all_user_prompts = []

        with pd.read_csv("data/wildchat/labels_50k.csv", chunksize=10000) as reader:
            for chunk in reader:
                for index, row in chunk.iterrows():
                    if int(row["Topic"]) == topic_id:
                        all_user_prompts.append(row["Document"])

        topic_prompts = random.sample(all_user_prompts, min(100, len(all_user_prompts)))

        for prompt in tqdm(topic_prompts, desc="Processing prompts"):
            prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
            file_path = Path("data/prompt_stats") / f"{prompt_hash}.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    assert json_data["prompt"] == prompt
                    
                    if "skywork-v2" not in json_data["summary_stats"]:
                        new_dict = {}
                        key_names = list(json_data["summary_stats"].keys())

                        for key in key_names:
                            val = json_data["summary_stats"][key]
                            new_dict[key] = val
                            del json_data["summary_stats"][key]
                        
                        json_data["summary_stats"]["skywork-v2"] = new_dict

                    else:
                        print("Already in correct format")
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=4)
            else:
                print("Prompt not found: ", prompt)
        
        # print("=" * 80)
        # print(f"Topic {topic_id}: {topic} with {len(all_user_prompts)} user prompts")
        # print("\nExample prompts:\n")
        # for prompt in all_user_prompts[:10]:
        #     print("-" * 80)
        #     print(prompt)


# %%
    # print(f"Sampling {len(prompts_to_sample)} prompts")

    # sample_responses(
    #     prompts=prompts_to_sample,
    #     policy_name="meta-llama/llama-3.1-8b-instruct",
    #     N=16,
    # )
# %%


# %%
