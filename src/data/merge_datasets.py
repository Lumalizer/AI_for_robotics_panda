import os
import shutil


def merge_datasets(dataset_names: list, new_dataset_name: str, raw_data_folder="datasets/raw_data"):
    npz_files = {}
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
            if filename.endswith(".npz"):
                original_ep_num = int(filename.split("episode_")[1].replace(".npz", ""))
                key = len(npz_files) + 1

                video_filename = filename.replace("npz", "mp4")
                main_camera_filename = video_filename.replace("episode", "primary_episode")
                wrist_filename = video_filename.replace("episode", "wrist_episode")

                npz_files[key] = f"{raw_data_folder}/{dataset_name}/{filename}"
                main_camera_files[key] = f"{raw_data_folder}/{dataset_name}/{main_camera_filename}"
                wrist_camera_files[key] = f"{raw_data_folder}/{dataset_name}/{wrist_filename}"
                csv_lines[key] = csv[original_ep_num]

    with open(f"{raw_data_folder}/{new_dataset_name}/data.csv", "w") as newcsv:
        newcsv.write("episode,task_description\n")

        for key in npz_files.keys():
            shutil.copy(npz_files[key], f"{raw_data_folder}/{new_dataset_name}/episode_{key}.npz")
            shutil.copy(main_camera_files[key], f"{raw_data_folder}/{new_dataset_name}/primary_episode_{key}.mp4")
            shutil.copy(wrist_camera_files[key], f"{raw_data_folder}/{new_dataset_name}/wrist_episode_{key}.mp4")
            newcsv.write(csv_lines[key])


if __name__ == "__main__":
    merge_datasets(["50_recover_from_extreme_positions_50", "100_stack_blue_red_100",
                    "100_stack_red_blue_100", "knock_over_bottle_25", "pack_box_50",
                    "pick_up_blue_200", "pick_up_bottle_25", "pick_up_doll_25",
                    "pick_up_doll_25_with_distractors", "pick_up_green_25_with_distractors",
                    "pick_up_red_100", "pick_up_sponge_25", "place_blue_in_box_50", "unpack_box_50"],
                   new_dataset_name="14datasets_05_12_2024_recover_stack_knockover_pack_pickup_place_unpack")
