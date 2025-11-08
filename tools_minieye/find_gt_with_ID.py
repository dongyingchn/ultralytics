import os
import json
from tqdm import tqdm

nas_dir_list = [
    "/mnt/nas111/alg_results/auto_label_results",
    "/mnt/nas111_1/alg_results/auto_label_results",
    "/mnt/nas111_2/alg_results/auto_label_results",
    "/mnt/pdcl/alg_results/auto_label_results",
]

records = {}

for nas_dir in nas_dir_list:

    vehicle_list = os.listdir(nas_dir)
    vehicle_list.sort()

    for vehicle_name in vehicle_list:
        if not vehicle_name.startswith("D4Q"):
            continue

        vehicle_dir = os.path.join(nas_dir, vehicle_name)

        date_list = os.listdir(vehicle_dir)
        date_list.sort()

        for date_name in date_list:

            date_dir = os.path.join(vehicle_dir, date_name)
            
            seq_list = os.listdir(date_dir)
            seq_list.sort()

            for seq_name in tqdm(seq_list, desc=f"Processing {nas_dir}/{vehicle_name}/{date_name}"):
                occ_gt_dir = os.path.join(date_dir, seq_name, "occ_gt", "bev")
                if os.path.exists(occ_gt_dir):
                    
                    if nas_dir not in records:
                        records[nas_dir] = {}
                    if vehicle_name not in records[nas_dir]:
                        records[nas_dir][vehicle_name] = {}
                    if date_name not in records[nas_dir][vehicle_name]:
                        records[nas_dir][vehicle_name][date_name] = []
                    records[nas_dir][vehicle_name][date_name].append(seq_name)

with open("gt_occ_bev_records.json", "w") as f:
    json.dump(records, f, indent=2)