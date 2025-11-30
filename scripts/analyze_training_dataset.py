import os
import glob
import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def gen_data_list():
    data_dir = "/mnt/mono3d/swji_data/Mono3d_4face_8m_d4q_bt601full"
    save_dir = "./train_data_v1/mono3d_D4Q"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cate_list = os.listdir(data_dir)
    cate_list.sort()
    for cate_name in cate_list:

        cate_dir = os.path.join(data_dir, cate_name)

        vehicle_list = os.listdir(cate_dir)
        vehicle_list.sort()
        for vehicle_name in vehicle_list:
            vehicle_dir = os.path.join(cate_dir, vehicle_name)
            date_list = os.listdir(vehicle_dir)
            date_list.sort()

            save_vehicle_dir = os.path.join(save_dir, cate_name, vehicle_name)
            os.makedirs(save_vehicle_dir, exist_ok=True)

            for date_name in tqdm(date_list, desc=f"Processing {cate_name}/{vehicle_name}"):
                date_dir = os.path.join(vehicle_dir, date_name)
                img_dir = os.path.join(date_dir, "images")

                img_list = os.listdir(img_dir)
                img_list.sort()
                img_path_list = [os.path.join(img_dir, img_name)+'\n' for img_name in img_list]
                
                save_date_file = os.path.join(save_vehicle_dir, f"{date_name}.txt")

                with open(save_date_file, 'w') as f:
                    f.writelines(img_path_list)

def get_data_statistics():

    data_list_dir = "./train_data_v1/mono3d_D4Q/data_list"
    cate_list = os.listdir(data_list_dir)
    cate_list.sort()

    total_frames = 0
    for cate_name in cate_list:
        cate_dir = os.path.join(data_list_dir, cate_name)
        vehicle_list = os.listdir(cate_dir)
        vehicle_list.sort()

        cate_frames = 0

        for vehicle_name in vehicle_list:
            vehicle_dir = os.path.join(cate_dir, vehicle_name)
            date_list = os.listdir(vehicle_dir)
            date_list.sort()
            
            for date_name in date_list:
                date_file = os.path.join(vehicle_dir, date_name)
                with open(date_file, 'r') as f:
                    lines = f.readlines()
                    cate_frames += len(lines)

        total_frames += cate_frames

        print(f"{cate_name} Frames: {cate_frames}")

    print(f"Total Frames: {total_frames}")

def parse_label_file(file_path):
    """解析单个标签文件并返回数据列表"""

    with open(file_path, 'r') as f:
        img_paths = f.readlines()

    txt_paths = [img_path.strip().replace('/images/', '/labels_dealmult/').replace('.jpg', '.txt') for img_path in img_paths]
    
    data = []
    for txt_path in tqdm(txt_paths, desc=f"解析文件 {os.path.basename(file_path)} 中的标签"):
    
        with open(txt_path, 'r') as f:
            for line in f:
                try:
                    words = line.strip().split()
                    parts = [words[0]] + [float(p) for p in words[1:]]
                    if len(parts) < 17:
                        label = parts[0]
                        data.append([label, np.nan, np.nan, np.nan, np.nan, '2D'])
                    
                    else:
                        # 根据您提供的格式进行解析
                        label = parts[0]
                        x3d_ori = parts[5]
                        z3d_ori = parts[7]
                        
                        rot_y = parts[12]
                        alpha_ori = parts[16] # alpha角度（弧度）

                        data.append([label, x3d_ori, z3d_ori, rot_y, alpha_ori, '3D'])
                except (ValueError, IndexError) as e:
                    # print(f"警告：跳过文件 '{txt_path}' 中的无效行: {line.strip()}，错误: {e}")
                    print(f"错误：{e}")
                    continue
    return data

def load_all_labels(label_dir):
    """加载所有标签文件到一个Pandas DataFrame中"""

    if os.path.isfile(label_dir):
        all_files = [label_dir]
    elif os.path.isdir(label_dir):
        all_files = glob.glob(os.path.join(label_dir, '*.txt'))
    if not all_files:
        raise FileNotFoundError(f"在目录 '{label_dir}' 中未找到任何 .txt 标签文件。")
        
    all_data = []
    # 使用tqdm显示处理进度
    for file_path in all_files:
        all_data.extend(parse_label_file(file_path))
    
    # 定义DataFrame的列名
    columns = ['label', 'x3d_ori', 'z3d_ori', 'rot_y', 'alpha_ori_rad', 'type']
    df = pd.DataFrame(all_data, columns=columns)
    
    # 添加类别名称和角度（度数）列
    df['class_name'] = df['label'] #.map(CONFIG['class_names'])
    df['alpha_ori_deg'] = df['alpha_ori_rad'].apply(lambda r: r * 180 / math.pi)

    # 填充未知的类别名称
    df['class_name'].fillna('Unknown', inplace=True)
    
    return df

def load_all_labels_from_list(file_list):
    """
    从一个给定的文本文件路径列表加载所有标签。
    这是 load_all_labels 的变体，直接接收文件列表。
    """
    if not file_list:
        return pd.DataFrame()
        
    all_data = []
    # 在这里为文件列表添加tqdm进度条
    for file_path in file_list:
        all_data.extend(parse_label_file(file_path))
    
    if not all_data:
        return pd.DataFrame()

    columns = ['label', 'x3d_ori', 'z3d_ori', 'rot_y', 'alpha_ori_rad', 'type']
    df = pd.DataFrame(all_data, columns=columns)
    
    df['class_name'] = df['label']
    df['alpha_ori_deg'] = df['alpha_ori_rad'].apply(lambda r: r * 180 / math.pi if pd.notna(r) else np.nan)
    df['class_name'].fillna('Unknown', inplace=True)
    
    return df

def chunk_list(data, chunk_size):
    """将一个列表切分成多个指定大小的小块"""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def cache_gt_statistics(chunk_size=10):
    data_list_dir = "./train_data_v1/mono3d_D4Q/data_list"
    cate_list = os.listdir(data_list_dir)
    cate_list.sort()

    cache_dir = "./train_data_v1/mono3d_D4Q/label_cache"

    for cate_name in cate_list:
        if cate_name in ["CNCAP", "driving"]:
            continue
        cate_dir = os.path.join(data_list_dir, cate_name)
        vehicle_list = os.listdir(cate_dir)
        vehicle_list.sort()

        for vehicle_name in vehicle_list:

            # if vehicle_name not in ['D4Q_51']:
            #     continue
            vehicle_dir = os.path.join(cate_dir, vehicle_name)

            # 找到该车辆下的所有 .txt 文件
            date_files = sorted(glob.glob(os.path.join(vehicle_dir, '*.txt')))
            
            if not date_files:
                continue

            # 准备缓存输出目录
            cache_vehicle_dir = os.path.join(cache_dir, cate_name, vehicle_name)
            os.makedirs(cache_vehicle_dir, exist_ok=True)
            
            # --- 核心改动：按 chunk_size 切分文件列表 ---
            file_chunks = list(chunk_list(date_files, chunk_size))
            
            # 遍历每个文件块
            for i, chunk in enumerate(file_chunks):
                # 为每个块生成一个独立的DataFrame
                # 注意：这里我们调用一个新函数，它直接处理文件列表
                print(f"Processing chunk {i+1}/{len(file_chunks)} for {cate_name}/{vehicle_name}...")
                df_chunk = load_all_labels_from_list(chunk)
                
                if df_chunk.empty:
                    continue

                chunk_start = chunk[0].split('/')[-1].replace('.txt', '')
                chunk_end = chunk[-1].split('/')[-1].replace('.txt', '')
                
                # 为每个块生成一个唯一的缓存文件名
                cache_file_name = f"{chunk_start}-{chunk_end}.pkl"
                    
                cache_file_path = os.path.join(cache_vehicle_dir, cache_file_name)
                
                # 保存这个块的DataFrame
                df_chunk.to_pickle(cache_file_path)

CONFIG = {
    # 存放 .pkl 缓存文件的根目录
    "cache_dir": "./train_data_v1/mono3d_D4Q/label_cache", 
    # 报表和图像的输出目录
    "output_dir": "./cache_data_profile_output", 
    # Z轴距离统计的配置
    "z_distribution": {
        "max_dist": 100,  # 分析的最大距离
        "interval": 10,   # 每个区间的宽度
    },
    "z_dist_target_classes": ["0", "1", "2", "3"] 
}
# --- 主配置区结束 ---


def load_all_cached_labels(cache_dir):
    """
    加载所有缓存的.pkl文件，并将它们合并成一个大的DataFrame。
    """
    all_pkl_files = glob.glob(os.path.join(cache_dir, '**', '*.pkl'), recursive=True)
    
    if not all_pkl_files:
        raise FileNotFoundError(f"在目录 '{cache_dir}' 及其子目录中未找到任何 .pkl 缓存文件。")
        
    df_list = []
    for pkl_path in tqdm(all_pkl_files, desc="正在加载缓存文件"):
        try:
            df = pd.read_pickle(pkl_path)
            df_list.append(df)
        except Exception as e:
            print(f"警告：加载文件 '{pkl_path}' 失败，错误: {e}")
            continue
    
    if not df_list:
        raise ValueError("未能成功加载任何有效的数据。")

    # 合并所有的DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def plot_z_distribution(df, output_dir, target_class=None):
    """
    对Z轴距离进行分箱统计，并绘制柱状图。
    """
    print("\n--- 正在统计Z轴距离分布 ---")
    
    # 1. 筛选出有3D信息的目标
    df_3d = df[df['type'] == '3D'].copy()

    if target_class:
        title_prefix = f'类别 "{target_class}" 的'
        filename_prefix = f'z_dist_{target_class}'
        df_3d = df_3d[df_3d['class_name'] == target_class]
    else:
        title_prefix = '所有3D目标的'
        filename_prefix = 'z_dist_overall'

    if df_3d.empty:
        print(f"没有找到类别为 '{target_class or 'Any'}' 的3D目标，跳过Z轴分布统计。")
        return

    print(f"--- 正在统计 {title_prefix} Z轴距离分布 ---")

    # 2. 准备分箱 (binning)
    max_dist = CONFIG["z_distribution"]["max_dist"]
    interval = CONFIG["z_distribution"]["interval"]
    bins = np.arange(0, max_dist + interval, interval)
    
    # 创建区间标签，例如 "[0, 10)", "[10, 20)"
    labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins)-1)]
    
    # 使用 pd.cut 对 z3d_ori 进行分箱
    df_3d['z_interval'] = pd.cut(df_3d['z3d_ori'], bins=bins, labels=labels, right=False)

    # 3. 统计每个区间的数量
    z_counts = df_3d['z_interval'].value_counts().sort_index()
    counts_df = z_counts.reset_index()
    counts_df.columns = ['z_interval', 'count']
    
    # 4. 开始绘图
    plt.figure(figsize=(18, 9))
    ax = sns.barplot(
        x='z_interval', 
        y='count', 
        data=counts_df,
        palette='plasma'
    )

    # 5. 在柱子上方添加文本标注
    total_3d_count = len(df_3d)
    for index, row in counts_df.iterrows():
        bar = ax.patches[index]
        percentage = (row['count'] / total_3d_count) * 100
        label = f"{row['count']}\n({percentage:.1f}%)"
        
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            row['count'], label, 
            ha='center', va='bottom', 
            fontsize=10, color='black'
        )

    # 6. 美化图表
    ax.set_title(f'3D objects z3d distribution (total: {total_3d_count})', fontsize=20, pad=20)
    ax.set_xlabel('distance interval (meter)', fontsize=14)
    ax.set_ylabel('number of objects', fontsize=14)
    ax.set_ylim(0, counts_df['count'].max() * 1.15)
    plt.xticks(rotation=45, ha='right')
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    sns.despine()

    # 7. 保存图像
    output_path = os.path.join(output_dir, f'{filename_prefix}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Z轴距离分布柱状图已保存至: {output_path}")


def main():
    """主函数，加载缓存并执行分析"""
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载所有缓存数据
        combined_df = load_all_cached_labels(CONFIG['cache_dir'])
        print(f"\n成功从缓存中加载 {len(combined_df)} 条总数据。")
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        return

    # 打印简要统计信息
    print("\n--- 整体目标类别统计 (2D + 3D) ---")
    print(combined_df['class_name'].value_counts().sort_index().to_string())

    type_counts = combined_df['type'].value_counts()
    print("\n--- 按类型细分 ---")
    print(f"3D 目标总数: {type_counts.get('3D', 0)}")
    print(f"纯2D 目标总数: {type_counts.get('2D', 0)}")
    
    # 调用Z轴距离分布统计函数
    plot_z_distribution(combined_df, output_dir)

    # 2. 为指定的几个类别，单独绘制Z轴分布图
    target_classes = CONFIG["z_dist_target_classes"]
    if not target_classes: # 如果列表为空，则为所有存在的3D类别生成
        target_classes = combined_df[combined_df['type'] == '3D']['class_name'].unique()
        
    for class_name in target_classes:
        plot_z_distribution(combined_df, output_dir, target_class=class_name)
    
    print("\n数据画像分析完成！")

if __name__ == "__main__":
    # gen_data_list()
    # get_data_statistics()
    # cache_gt_statistics()

    main()