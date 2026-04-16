from datasets import load_from_disk
which_datasets = ["robomimic_eval",]
for dataset in which_datasets:
    ds = load_from_disk(f"robometer_dataset/robomimic_rbm/{dataset}")
    ds.push_to_hub("robomimic_rbm_eval", dataset)

from huggingface_hub import HfApi
api = HfApi()
api.upload_large_folder(folder_path='robometer_dataset/robomimic_rbm/robomimic_eval', repo_id='robomimic_rbm_eval', repo_type='dataset', ignore_patterns=["*data*.arrow", "*state.json", "*info.json"])