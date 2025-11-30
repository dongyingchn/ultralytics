import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math

# plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# # This line is crucial to properly display negative signs (-)
# plt.rcParams['axes.unicode_minus'] = False 

# --- 配置区 ---
CONFIG = {
    "label_dir": "/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/train_data/D4Q/train_all.txt",  # 存放真值 .txt 文件的目录
    "output_dir": "./train_data/D4Q/data_profile_output", # 报表和图像的输出目录
    "cache_filename": "parsed_labels_cache.pkl",

    "class_names": ["car",  "tinycar", "bus", "van", "truck","tanker", "large_truck", "construction_vehicle","special_vehicle", "unknown", # 0-9
                    "pedestrian", "bicycle", "bicyclist", # 10-12
                    "motorcycle", "motorcyclist", "tricycle", "tricyclist", # 13-16
                    'traffic_light_bbox', 'traffic_light_bulb',
                    'traffic_sign', 'animal', 'movable_object',
                    'warning_triangle', 'traffic_cone', 'water_barrier', 'crash_barrel',
                    'movable_barrier', 'bollard', 'sphere_bollard', 'cube_bollard',
                    'cylinder_bollard', 'construction_barrier', 'other_barrier',
                    'road_barrier_unknown', "wheel", "plate", "face"], 
    # "class_names": {  # 类别ID到名称的映射，请按需修改
    #     0: "Car",
    #     1: "Pedestrian",
    #     2: "Cyclist",
    #     3: "Truck",
    #     4: "Van",
    #     # ... 在这里添加更多类别
    # },
    "position_map_range": { # 位置分布图的坐标范围 (米)
        "x_min": -50,
        "x_max": 50,
        "z_min": 0,
        "z_max": 100,
    }
}
# --- 配置区结束 ---


def parse_label_file(file_path):
    """解析单个标签文件并返回数据列表"""

    with open(file_path, 'r') as f:
        img_paths = f.readlines()

    txt_paths = [img_path.strip().replace('/images/', '/labels/').replace('.jpg', '.txt') for img_path in img_paths]
    
    data = []
    for txt_path in tqdm(txt_paths, desc=f"解析文件 {os.path.basename(file_path)} 中的标签"):
    
        with open(txt_path, 'r') as f:
            for line in f:
                try:
                    words = line.strip().split()
                    parts = [words[0]] + [float(p) for p in words[1:]]
                    if len(parts) < 17:
                        label = parts[0]
                        data.append([label, np.nan, np.nan, np.nan, '2D'])
                    
                    else:
                        # 根据您提供的格式进行解析
                        label = parts[0]
                        x3d_ori = parts[5]
                        z3d_ori = parts[7]
                        
                        # rot_y = parts[12]
                        alpha_ori = parts[16] # alpha角度（弧度）

                        data.append([label, x3d_ori, z3d_ori, alpha_ori, '3D'])
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
    columns = ['label', 'x3d_ori', 'z3d_ori', 'alpha_ori_rad', 'type']
    df = pd.DataFrame(all_data, columns=columns)
    
    # 添加类别名称和角度（度数）列
    df['class_name'] = df['label'] #.map(CONFIG['class_names'])
    df['alpha_ori_deg'] = df['alpha_ori_rad'].apply(lambda r: r * 180 / math.pi)

    # 填充未知的类别名称
    df['class_name'].fillna('Unknown', inplace=True)
    
    return df


def plot_position_distribution(df, output_dir):
    """为每个类别绘制位置分布的鸟瞰图"""
    print("\n--- 正在生成位置分布图 ---")
    
    unique_labels = df['label'].unique()
    
    for label_id in tqdm(unique_labels, desc="生成位置图中"):
        # class_name = CONFIG['class_names'].get(label_id, f"Unknown_{label_id}")
        class_df = df[df['label'] == label_id]
        
        if class_df.empty:
            continue

        plt.figure(figsize=(10, 12))
        
        # 使用 seaborn 的 2D 核密度估计图 (kdeplot) 来创建热力图效果
        # sns.kdeplot(
        #     data=class_df, 
        #     x='x3d_ori', 
        #     y='z3d_ori', 
        #     fill=True, 
        #     thresh=0.05, 
        #     cmap='viridis',
        #     cbar=True
        # )
        
        # 也可以使用散点图，如果目标数量不多的话
        plt.scatter(class_df['x3d_ori'], class_df['z3d_ori'], s=1, alpha=0.5)

        plt.title(f'"{label_id}", number: {len(class_df)}', fontsize=16)
        plt.xlabel('X (meter)', fontsize=12)
        plt.ylabel('Z (meter)', fontsize=12)
        plt.xlim(CONFIG["position_map_range"]["x_min"], CONFIG["position_map_range"]["x_max"])
        plt.ylim(CONFIG["position_map_range"]["z_min"], CONFIG["position_map_range"]["z_max"])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.gca().set_aspect('equal', adjustable='box') # 保持坐标轴比例一致
        
        # 在(0,0)位置标注车辆自身
        plt.plot(0, 0, 'r^', markersize=12)
        plt.legend()
        
        output_path = os.path.join(output_dir, f'position_distribution_{label_id}.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
    print("位置分布图已保存至:", output_dir)


def plot_angle_distribution(df, output_dir):
    """为每个类别绘制alpha角度分布直方图"""
    print("\n--- 正在生成角度分布图 ---")
    
    unique_labels = df['label'].unique()
    
    for label_id in tqdm(unique_labels, desc="生成角度图中"):
        # class_name = CONFIG['class_names'].get(label_id, f"Unknown_{label_id}")
        class_df = df[df['label'] == label_id]

        if class_df.empty:
            continue

        plt.figure(figsize=(12, 6))
        
        # 绘制直方图，分为72个箱子（每5度一个）
        sns.histplot(class_df['alpha_ori_deg'], bins=72, kde=True, color='skyblue')
        
        plt.title(f'"{label_id}", number: {len(class_df)}', fontsize=16)
        plt.xlabel('Alpha', fontsize=12)
        plt.ylabel('number', fontsize=12)
        plt.xticks(np.arange(-180, 181, 30)) # 设置x轴刻度
        plt.grid(True, linestyle='--', alpha=0.6)
        
        output_path = os.path.join(output_dir, f'angle_distribution_{label_id}.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
    print("角度分布图已保存至:", output_dir)

def plot_polar_distribution(df, output_dir):
    """
    为每个类别绘制深度和角度的极坐标散点图。
    - 角度 (alpha_ori) 决定点在圆上的位置。
    - 深度 (z3d_ori) 决定点离圆心的距离。
    """
    print("\n--- 正在生成极坐标分布图 ---")

    unique_labels = df['label'].unique()

    for label_id in tqdm(unique_labels, desc="生成极坐标图中"):
        # class_name = CONFIG['class_names'].get(label_id, f"Unknown_{label_id}")
        class_df = df[df['label'] == label_id]

        if class_df.empty:
            continue

        # 创建一个使用极坐标投影的子图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

        # 提取数据
        # 在极坐标中，theta是角度，r是半径
        theta = class_df['alpha_ori_rad']  # 角度使用弧度制
        r = class_df['z3d_ori']            # 半径使用深度/距离

        # 绘制散点图
        # c=r: 根据距离(r)来着色，使远处的点和近处的点颜色不同
        # cmap='coolwarm': 使用冷暖色调，可以直观看出距离变化
        # alpha=0.6: 设置透明度，以缓解点重叠的问题
        scatter = ax.scatter(theta, r, c=r, cmap='coolwarm', alpha=0.6, s=10)

        # --- 美化图形 ---
        # 设置图表标题
        ax.set_title(f'"{label_id}", number: {len(class_df)}', va='bottom', fontsize=16)

        # 设置半径(r)轴的标签和范围
        ax.set_ylabel("depth/distance (meter)", labelpad=20, fontsize=12)
        # ax.set_rmax(CONFIG["position_map_range"]["z_max"]) # 可以设置最大半径
        ax.set_rmax(class_df['z3d_ori'].max() * 1.1) # 根据数据动态设置最大半径
        ax.set_rlabel_position(-22.5)  # 移动半径标签位置，防止重叠

        # 设置角度(theta)轴的标签
        # Matplotlib极坐标默认0度在右侧，逆时针增加。这与车辆前方为0度的习惯不同。
        # 我们可以调整它以匹配习惯。0度朝上。
        # 设置默认角度0度在右侧
        ax.set_theta_zero_location('E')  # 将0度设置在东方（右侧）
        # ax.set_theta_zero_location('N')  # 将0度设置在北方（上方）
        ax.set_theta_direction(-1)       # 将角度方向设置为顺时针

        # 添加颜色条 (colorbar) 来解释颜色与距离的对应关系
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
        cbar.set_label('depth/distance (meter)', fontsize=12)

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.6)

        # 保存图像
        output_path = os.path.join(output_dir, f'polar_distribution_{label_id}.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
    print("极坐标分布图已保存至:", output_dir)

def plot_pie_charts(df, output_dir):
    """
    为2D和3D目标分别生成类别分布的饼图。
    """
    print("\n--- 正在生成类别分布饼图 ---")

    # 定义一个内部函数来绘制单个饼图，避免代码重复
    def _create_pie(data, title, filename, max_slices=50):
        if data.empty:
            print(f"没有数据可用于生成饼图: {title}")
            return
            
        class_counts = data['class_name'].value_counts()

        # 如果类别过多，将数量少的合并为“其他”
        if len(class_counts) > max_slices:
            others = class_counts[max_slices-1:].sum()
            class_counts = class_counts[:max_slices-1]
            class_counts['其他'] = others
        
        # 准备绘图数据
        labels = class_counts.index
        sizes = class_counts.values
        total = sizes.sum()

        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 使用好看的颜色主题
        colors = plt.cm.Pastel2(np.arange(len(labels)))
        
        # 绘制饼图
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            autopct=lambda p: f'{p:.1f}%\n({int(round(p * total / 100.0))})', # 显示百分比和具体数量
            startangle=90, 
            colors=colors,
            pctdistance=0.85, # 百分比文本离圆心的距离
            textprops={'fontsize': 12},
            wedgeprops=dict(width=0.4, edgecolor='w') # "甜甜圈"效果
        )
        
        # 设置标题
        ax.set_title(title, fontsize=18, pad=20)
        
        # 确保饼图是圆的
        ax.axis('equal')  

        # 添加图例
        ax.legend(wedges, labels,
                  title="类别",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))

        plt.setp(autotexts, size=10, weight="bold", color="black")
        
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

    # 筛选数据并为2D和3D目标分别调用绘图函数
    _create_pie(df, '2D objects', 'pie_chart_2d_distribution.png')

    print("类别分布饼图已保存至:", output_dir)

def plot_pie_chart_overall_v1(df, output_dir, max_slices=20, drilldown_threshold=5):
    """
    为数据集中所有目标生成一个总的类别分布饼图。
    如果存在合并的“其他”类别，则为其额外生成一个下钻式子饼图。
    """
    print("\n--- 正在生成整体类别分布饼图 ---")

    if df.empty:
        print("没有数据可用于生成饼图。"); return
        
    class_counts = df['class_name'].value_counts()
    total_count = len(df)
    
    # --- 主饼图数据准备 ---
    main_counts = class_counts.copy()
    others_data = None
    
    if len(class_counts) > max_slices:
        others_data = class_counts[max_slices-1:]
        others_sum = others_data.sum()
        main_counts = class_counts[:max_slices-1]
        if others_sum > 0:
            main_counts['other'] = others_sum

    # --- 内部函数：绘制饼图 ---
    def _draw_pie(counts_data, total_ref, title, filename, is_drilldown=False):
        labels = counts_data.index
        sizes = counts_data.values
        
        fig, ax = plt.subplots(figsize=(12, 10))
        colors = plt.cm.Pastel2(np.arange(len(labels))) if not is_drilldown else plt.cm.Set3(np.arange(len(labels)))
        
        # 核心改动：autopct现在总是基于 total_ref (总数据集大小) 来计算百分比
        def get_autopct(p):
            absolute_count = int(round(p * sizes.sum() / 100.0))
            # 显示相对于总数据集的百分比
            overall_percent = (absolute_count / total_ref) * 100
            return f'{overall_percent:.1f}%\n({absolute_count})'

        wedges, texts, autotexts = ax.pie(
            sizes, 
            autopct=get_autopct,
            startangle=90, 
            colors=colors,
            pctdistance=0.80,
            textprops={'fontsize': 12},
            wedgeprops=dict(width=0.4, edgecolor='w')
        )
        
        ax.set_title(title, fontsize=18, pad=20)
        ax.axis('equal')
        ax.legend(wedges, labels, title="cate", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=10, weight="bold", color="black")
        
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

    # 1. 绘制主饼图
    _draw_pie(main_counts, total_count, '2D + 3D', 'pie_chart_overall_distribution.png')
    
    # 2. 如果存在“其他”类别，则为其绘制下钻式饼图
    if others_data is not None and not others_data.empty:
        # 如果“其他”中的类别数量仍然很多，可以再次合并
        # if len(others_data) > drilldown_threshold:
        #      drilldown_others_sum = others_data[drilldown_threshold-1:].sum()
        #      others_data = others_data[:drilldown_threshold-1]
        #      if drilldown_others_sum > 0:
        #         others_data['更少类别'] = drilldown_others_sum

        print("检测到占比较少的类别，正在生成'其他'类别的下钻饼图...")
        _draw_pie(
            others_data, 
            total_count,  # 仍然使用总数作为参考
            "'others'", 
            'pie_chart_others_drilldown.png',
            is_drilldown=True
        )

    print("类别分布饼图已保存至:", output_dir)

def plot_pie_chart_overall(df, output_dir, max_slices=20, drilldown_threshold=5):
    """
    为数据集中所有目标生成一个总的类别分布饼图，标签显示在外部。
    如果存在合并的“其他”类别，则为其额外生成一个下钻式子饼图。
    """
    print("\n--- 正在生成整体类别分布饼图 ---")

    if df.empty:
        print("没有数据可用于生成饼图。"); return
        
    class_counts = df['class_name'].value_counts()
    total_count = len(df)
    
    main_counts = class_counts.copy()
    others_data = None
    
    if len(class_counts) > max_slices:
        others_data = class_counts[max_slices-1:]
        others_sum = others_data.sum()
        main_counts = class_counts[:max_slices-1]
        if others_sum > 0:
            main_counts['others'] = others_sum

    # --- 内部函数：绘制饼图（采用外部标签） ---
    def _draw_pie_external_labels(counts_data, total_ref, title, filename, is_drilldown=False):
        labels = counts_data.index
        sizes = counts_data.values
        
        fig, ax = plt.subplots(figsize=(14, 11), subplot_kw=dict(aspect="equal"))
        colors = plt.cm.Pastel2(np.arange(len(labels))) if not is_drilldown else plt.cm.Set3(np.arange(len(labels)))
        
        # 1. 绘制饼图，但这次不显示内部标签 (autopct) 和外部标签 (labels)
        wedges, texts = ax.pie(
            sizes, 
            colors=colors,
            startangle=90, 
            wedgeprops=dict(width=0.4, edgecolor='w') # 甜甜圈效果
        )

        # 2. 手动绘制外部标签和引导线
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1 # 计算扇区的中间角度
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            
            # 设置连接线的起始位置
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            
            # 准备标签文本
            absolute_count = counts_data.iloc[i]
            overall_percent = (absolute_count / total_ref) * 100
            label_text = f"{labels[i]}\n{overall_percent:.1f}% ({absolute_count})"
            
            # 绘制注释
            ax.annotate(label_text, xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                        horizontalalignment=horizontalalignment, **kw, fontsize=12)

        ax.set_title(title, fontsize=20, pad=20)
        
        # 保存图像
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

    # 1. 绘制主饼图
    _draw_pie_external_labels(main_counts, total_count, '2D + 3D', 'pie_chart_overall_distribution.png')
    
    # 2. 如果存在“其他”类别，则为其绘制下钻式饼图
    if others_data is not None and not others_data.empty:
        # if len(others_data) > drilldown_threshold:
        #      drilldown_others_sum = others_data[drilldown_threshold-1:].sum()
        #      others_data = others_data[:drilldown_threshold-1]
        #      if drilldown_others_sum > 0:
        #         others_data['更少类别'] = drilldown_others_sum

        print("检测到占比较少的类别，正在生成'其他'类别的下钻饼图...")
        _draw_pie_external_labels(
            others_data, 
            total_count,
            "'others'", 
            'pie_chart_others_drilldown.png',
            is_drilldown=True
        )

    print("类别分布饼图已保存至:", output_dir)

def plot_barplot_distribution(df, output_dir):
    """
    为数据集中所有目标生成一个按数量排序的类别分布柱状图。
    每个柱子上方会标注该类别的具体数量和在总数据集中的占比。
    """
    print("\n--- 正在生成类别分布柱状图 ---")

    if df.empty:
        print("没有数据可用于生成柱状图。")
        return

    # 1. 准备数据
    class_counts = df['class_name'].value_counts()
    total_count = len(df)
    
    # 将Series转换为DataFrame，方便seaborn使用
    counts_df = class_counts.reset_index()
    counts_df.columns = ['class_name', 'count']
    
    # 计算百分比
    counts_df['percentage'] = (counts_df['count'] / total_count) * 100

    # 2. 开始绘图
    plt.figure(figsize=(16, 9))
    
    # 使用seaborn绘制柱状图，它会自动排序并提供美观的颜色
    ax = sns.barplot(
        x='class_name', 
        y='count', 
        data=counts_df,
        palette='viridis' # 使用一个好看的调色板
    )

    # 3. 在每个柱子上方添加文本标注
    for index, row in counts_df.iterrows():
        # 获取柱子的中心x坐标
        bar = ax.patches[index]
        # 准备标注文本：数量和百分比
        label = f"{row['count']}\n({row['percentage']:.1f}%)"
        # 使用ax.text进行标注
        ax.text(
            bar.get_x() + bar.get_width() / 2, # x坐标
            row['count'],                      # y坐标
            label,                             # 文本内容
            ha='center',                       # 水平居中对齐
            va='bottom',                       # 垂直底部对齐
            fontsize=11,
            color='black'
        )

    # 4. 美化图表
    ax.set_title('2D + 3D', fontsize=20, pad=20)
    ax.set_xlabel('class', fontsize=14)
    ax.set_ylabel('number', fontsize=14)
    
    # 调整y轴范围，给顶部的文本留出空间
    ax.set_ylim(0, counts_df['count'].max() * 1.15)
    
    # 旋转x轴标签，以防类别名称过长导致重叠
    plt.xticks(rotation=45, ha='right')
    
    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

    # 移除图表的顶部和右侧边框，使其更简洁
    sns.despine()

    # 5. 保存图像
    output_path = os.path.join(output_dir, 'barplot_overall_distribution.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print("类别分布柱状图已保存至:", output_dir)

def main():
    """主函数，执行所有分析和绘图任务"""
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建缓存文件的完整路径
    cache_path = os.path.join(output_dir, CONFIG['cache_filename'])

    # --- 新增的缓存判断逻辑 ---
    if os.path.exists(cache_path):
        print(f"发现缓存文件 '{cache_path}'，正在直接加载...")
        try:
            df = pd.read_pickle(cache_path)
            print("从缓存加载DataFrame成功！")
        except Exception as e:
            print(f"从缓存加载失败: {e}。将重新解析原始文件。")
            df = load_all_labels(CONFIG['label_dir'])
            df.to_pickle(cache_path)
            print(f"已重新解析并将DataFrame缓存至 '{cache_path}'。")
    else:
        print("未发现缓存文件，开始解析原始标签文件...")
        try:
            df = load_all_labels(CONFIG['label_dir'])
            # 将新处理的DataFrame保存到缓存文件
            df.to_pickle(cache_path)
            print(f"解析完成，并将DataFrame缓存至 '{cache_path}'。")
        except FileNotFoundError as e:
            print(f"错误: {e}")
            return
    # --- 缓存逻辑结束 ---

    if df.empty:
        print("未加载到任何有效数据，程序退出。")
        return
        
    # 1. 统计和打印类别数量
    # 筛选出3D和2D目标
    df_3d = df[df['type'] == '3D']
    df_2d = df[df['type'] == '2D']

    print("\n--- 3D目标类别统计 ---")
    if not df_3d.empty:
        print(df_3d['class_name'].value_counts().sort_index().to_string())
    else:
        print("数据集中没有发现3D目标。")
        
    print("\n--- 纯2D目标类别统计 ---")
    if not df_2d.empty:
        print(df_2d['class_name'].value_counts().sort_index().to_string())
    else:
        print("数据集中没有发现纯2D目标。")

    print("\n--- 全部目标类别统计 ---")
    print(df['class_name'].value_counts().sort_index().to_string())
    
    # 2. 绘制位置分布图
    # plot_position_distribution(df_3d, output_dir)
    
    # # 3. 绘制角度分布图
    # plot_angle_distribution(df_3d, output_dir)

    # plot_polar_distribution(df_3d, output_dir)

    # plot_pie_charts(df, output_dir)
    # plot_pie_chart_overall_v1(df, output_dir)
    plot_barplot_distribution(df, output_dir)
    
    print("\n数据画像分析完成！")

if __name__ == '__main__':
    main()