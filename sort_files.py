import os
import shutil
import pandas as pd

# sets the dataset
dataset = "p2_6wpi"
# file_name = f"{dataset}_data_notes.csv"
# csv_file = os.path.join(
#         "/home/jhuang/Documents/phd_projects/injected_GC_data/tables",
#         dataset,
#         file_name,
#     )

# # gets unique cell names
# sweep_info = pd.read_csv(csv_file, index_col=0)

# cell_names_list = []

# for file in sweep_info['File Path']:
#     split = file.split("_")
#     cell_name = f"{split[0]}_{split[1]}"
#     if not cell_name in cell_names_list:
#         cell_names_list.append(cell_name)

# # makes directories for each file
# base_path = f"/home/jhuang/Documents/phd_projects/scored_GC_data/{dataset}/"

# for cell_name in cell_names_list:
#     directory = os.path.join(base_path, cell_name)
#     print(directory)

# this organizes the ibw files to be scored

base_path = (
    f"/home/jhuang/Documents/phd_projects/scored_GC_data/original/{dataset}/"
)

os.chdir(base_path)

for file in os.listdir(base_path):
    split = file.split("_")
    cell_name = f"{split[0]}_{split[1]}"

    if not os.path.exists(cell_name):
        os.mkdir(cell_name)
        shutil.copy(os.path.join(base_path, file), cell_name)
    else:
        shutil.copy(os.path.join(base_path, file), cell_name)


# this gets the ibw file names for data_notes
ibw_file_list = []
cell_name_list = []

for cell_name in os.listdir(base_path):
    cell_name_list.extend([[cell_name] * 2])
    os.chdir(os.path.join(base_path, cell_name))
    for file in os.listdir(os.path.join(base_path, cell_name)):
        if file.endswith(".ibw"):
            ibw_file_list.append(file)
print(ibw_file_list)

ibw_file_list = pd.DataFrame(ibw_file_list)

flatten_cell_list = [cell for sublist in cell_name_list for cell in  sublist]
cell_name_list = pd.DataFrame(flatten_cell_list)

data_notes_cells = pd.concat([ibw_file_list, cell_name_list], axis=1)

data_notes_cells.to_csv("cells.csv")


# this moves mod output files to the proper folder in injected_GC_data

mod_output = (
    f"/home/jhuang/Documents/phd_projects/mod-2.0/identified_events/{dataset}/"
)
mod_folder = f"/home/jhuang/Documents/phd_projects/injected_GC_data/mod_events/{dataset}/"

os.chdir(mod_output)

if not os.path.exists(mod_folder):
    os.mkdir(mod_folder)

for file in os.listdir(mod_output):
    split = file.split("_")
    cell_name = f"{split[0]}_{split[1]}"
    light_events = f"{cell_name}_light100.mod.w4.e1.h13.minidet.mat"
    spontaneous_events = f"{cell_name}_spontaneous.mod.w4.e1.h13.minidet.mat"

    light_source = os.path.join(mod_output, file, light_events)
    light_dest = os.path.join(mod_folder, light_events)

    spontaneous_source = os.path.join(mod_output, file, spontaneous_events)
    spontaneous_dest = os.path.join(mod_folder, spontaneous_events)

    shutil.copy(light_source, light_dest)
    shutil.copy(spontaneous_source, spontaneous_dest)

