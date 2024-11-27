import os
import shutil

def merge_datasets(dataset_names: list, new_dataset_name: str, raw_data_folder="datasets/raw_data"):
    pickle_files = {}
    main_camera_files = {}
    wrist_camera_files = {}
    csv_lines = {}
    
    for dataset_name in dataset_names:
        if not os.path.exists(f"{raw_data_folder}/{dataset_name}"):
            raise Exception(f"Dataset {dataset_name} does not exist.")    
       
    if not os.path.exists(f"{raw_data_folder}/{new_dataset_name}"):
        os.makedirs(f"{raw_data_folder}/{new_dataset_name}")
        
    for dataset_name in dataset_names:
        with open(f"{raw_data_folder}/{dataset_name}/data.csv", "r") as f:
            csv = f.readlines()
        for filename in os.listdir(f"{raw_data_folder}/{dataset_name}"):
            if filename.endswith(".pkl"):
                original_ep_num = int(filename.split("episode_")[1].replace(".pkl", ""))
                key = len(pickle_files) + 1
                
                video_filename = filename.replace("pkl", "mp4")
                main_camera_filename = video_filename.replace("episode", "primary_episode")
                wrist_filename = video_filename.replace("episode", "wrist_episode")
                
                pickle_files[key] = f"{raw_data_folder}/{dataset_name}/{filename}"
                main_camera_files[key] = f"{raw_data_folder}/{dataset_name}/{main_camera_filename}"
                wrist_camera_files[key] = f"{raw_data_folder}/{dataset_name}/{wrist_filename}"
                csv_lines[key] = csv[original_ep_num]
                
    with open(f"{raw_data_folder}/{new_dataset_name}/data.csv", "w") as newcsv:
        newcsv.write("episode,task_description\n")
                
        for key in pickle_files.keys():
            shutil.copy(pickle_files[key], f"{raw_data_folder}/{new_dataset_name}/episode_{key}.pkl")
            shutil.copy(main_camera_files[key], f"{raw_data_folder}/{new_dataset_name}/primary_episode_{key}.mp4")
            shutil.copy(wrist_camera_files[key], f"{raw_data_folder}/{new_dataset_name}/wrist_episode_{key}.mp4")
            newcsv.write(csv_lines[key])
    
   
if __name__ == "__main__":
    merge_datasets(["octo_with_wrist_RAW_diagnostic_close", 
                    "octo_with_wrist_RAW_diagnostic_wide"], 
                   new_dataset_name="grasp_blue_300_red_100")