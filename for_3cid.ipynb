import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
import shutil
import cv2
import matplotlib.pyplot as plt
import easyocr
import sys
import random 
from PIL import Image
import os
from colorthief import ColorThief
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from skimage import color
import pandas as pd
import matplotlib.pyplot as plt


# 确保你已经安装了必要的库，并且路径正确
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# 定义基础路径
base_path = '/home/heyunshen/code/data/background_color/珠宝首饰'
# base_path = '/home/heyunshen/code/data/time/图书'

# 读取基础路径下的所有子文件夹
subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

# 加载训练后的YOLOv8模型
model = YOLO(r'/home/heyunshen/code/model/yolov8l-seg.pt', task='segment')

# 加载模型和配置
sam_checkpoint = "/home/heyunshen/code/model/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
# device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# 加载OCR识别模型
reader = easyocr.Reader(['ch_sim', 'en'])



# 循环遍历每个子文件夹
for folder in subfolders:
    # data_path = os.path.join(folder, '智能背景_0to200_0412.csv')  # 数据地址
    folder_path = os.path.join(folder)  # 原始图保存地址
    folder_a = os.path.join(folder, 'img_ok/')  # 在yolo中result不为null的图
    folder_b = os.path.join(folder, 'img_null/')  # 在yolo中result为null的图
    input_folder = os.path.join(folder, 'img_ok/')
    output_folder = os.path.join(folder, 'img_crop')  # 分割后的图
    # folder_A = os.path.join(folder, 'img_crop')
    folder_B = os.path.join(folder, 'img_color')  # 分割后的非白底图
    folder_C = os.path.join(folder, 'img_white')  # 分割后的白底图
    output_folder_A = os.path.join(folder, 'img_color_canny_a')  # 纯色图
    output_folder_B = os.path.join(folder, 'img_color_canny_b')  # 复杂场景图
    output_folder_C = os.path.join(folder, 'img_color_canny_c')  # 黑色像素过多的图(分割失误造成的)
    output_file = os.path.join(folder, 'img_color_canny_a', 'color_main.xlsx')  # 主色结果的Excel
    # cropped_path = base_path + 'cropped_image.png'  # 裁剪后的图片
    cropped_path = os.path.join(folder, 'cropped_image.png')  # 主色结果的Excel



    # n = 0.25  # 白底图筛选阈值
    black_pixel_threshold = 0.85  # 黑色像素点占比阈值
    canny_low_threshold = 30  # canny筛选阈值
    canny_high_threshold = 50  # canny筛选阈值
    threshold_percentage = 0.02  # 复杂度筛选阈值(canny像素分拣)

    # 确保你已经安装了必要的库，并且路径正确
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamPredictor



    '''
    result是否为null检测
    '''
    # # 加载训练后的YOLOv8模型
    # model = YOLO(r'd://code//yolo//yolov8l-seg.pt', task='segment')

    # 图片源文件夹和目标文件夹路径
    source_folder = folder_path

    # 确保目标文件夹存在
    os.makedirs(folder_a, exist_ok=True)
    os.makedirs(folder_b, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(source_folder, filename)
            
            # 使用YOLO模型进行预测
            results = model.predict(image_path)
            
            if len(results) > 0:
                result = results[0]
                
                if len(result.boxes) > 0:
                    box = result.boxes[0]
                    class_id = box.cls[0].item()
    #                 print(filename, 'yes')

                    # 将图片复制到文件夹A
                    shutil.copy(image_path, os.path.join(folder_a, filename))
                    # print(filename, '复制到img_ok')

                else:
                    shutil.copy(image_path, os.path.join(folder_b, filename))
                    # print(filename, '复制到img_null')
    print('result检测完成')



    '''
    删除贴片和主体
    '''
    # # 加载模型和配置
    # sam_checkpoint = "D://aigc//ComfyUI_windows_portable//ComfyUI//models//sams//sam_vit_h_4b8939.pth"
    # model_type = "vit_h"
    # device = "cuda"

    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)

    # predictor = SamPredictor(sam)

    # # 加载OCR识别模型
    # reader = easyocr.Reader(['ch_sim', 'en'])

    # # Load a model
    # model = YOLO(r'd://code//yolo//yolov8l-seg.pt', task='segment')

    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有图片
    # for image_filename in os.listdir(input_folder):
    for image_filename in tqdm(os.listdir(input_folder), desc='Processing images'):

        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(input_folder, image_filename)
            
            # 读取图片
            image = cv2.imread(image_path)
            
            if image is None:
                continue

            image_p = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 转换图像为RGB模式
            cropped_image = Image.fromarray(image_p).convert('RGB')
            image_np = np.array(cropped_image)
            
            # 进行OCR识别
            result = reader.readtext(image_np)
            
            # 提取所有坐标点
            all_coordinates = []
            for item in result:
                coordinates = item[0]
                for coord in coordinates:
                    all_coordinates.append(coord)
            
            # 重新组合坐标格式
            reformatted_coordinates = [[coord[0], coord[1]] for coord in all_coordinates]
            # reformatted_coordinates = [coord for coord in reformatted_coordinates if coord[1] > 200]
        
            # 初始化修改后的图片
            modified_image = image.copy()
            
            # 如果reformatted_coordinates为空数组，直接进入YOLO过程
            if not reformatted_coordinates:
                results = model.predict(image_np, show=False)
                if len(results) > 0:
                    result = results[0]
                    if len(result.boxes) > 0:
                        for result in results:
                            masks = result.masks.cpu().data.numpy()

                            for mask in masks:
                                mask_binary = (mask > 0.5).astype(np.uint8)
                                mask_binary_resized = cv2.resize(mask_binary, (image.shape[1], image.shape[0]), 
                                                                interpolation=cv2.INTER_NEAREST)
                                mask_indices = np.where(mask_binary_resized == 1)
                                modified_image[mask_indices[0], mask_indices[1]] = [0, 0, 0]
                                
                                # 保存修改后的图片
                                output_filename = f"{os.path.splitext(image_filename)[0]}.png"
                                cv2.imwrite(os.path.join(output_folder, output_filename), modified_image)       
            else:
                image_x = cv2.imread(image_path)

                image_p_x = image_x
                image_p_x = cv2.cvtColor(image_p_x, cv2.COLOR_BGR2RGB)  

                predictor.set_image(image_p_x)

                n = len(reformatted_coordinates)
                my_list = [1] * n

                input_point_p = np.array(reformatted_coordinates)
                input_label_p = np.array(my_list)

                masks, scores, logits = predictor.predict(
                    point_coords=input_point_p,
                    point_labels=input_label_p,
                    multimask_output=True,
                )

                masks.shape  # (number_of_masks) x H x W


                # 这里是将segment所有的mask合并起来, 相当于取了一个交集输出, 最大限度的保留主体的识别区
                all_masks = []
                for mask, score in zip(masks, scores):

                    # 对于每个掩码mask，将非零像素设置为1
                    binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)

                    # 将掩码乘以得分，以加权合并掩码
                    weighted_mask = binary_mask * score
                    all_masks.append(weighted_mask)

                # 将所有加权掩码加起来以获得最终掩码
                merged_mask = np.sum(all_masks, axis=0)

                # 可选：将二进制图像转回掩码形式
                final_mask = np.where(merged_mask > 0, 1, 0).astype(np.uint8)

                # 显示最终掩码
        #         show_mask(final_mask, plt.gca())

                # 将最终掩码转换为 PIL 图像
                mask_image = Image.fromarray(final_mask * 255)

                # 将原图像转换为 PIL 图像
                original_image = Image.fromarray(image_p)

                # 创建一个新的 RGBA 图像，将原图像复制到红、绿、蓝通道，将掩码图像复制到 Alpha 通道
                cropped_image = Image.new("RGBA", original_image.size)
                cropped_image.paste(original_image, (0, 0))

                # 在掩码区域上应用透明度（Alpha 通道）
                cropped_image.paste((0, 0, 0, 0), mask=mask_image)

                # 保存裁剪后的图像为 PNG 格式
                # cropped_image.save('D://code//data//1.28//cropped_image.png', format='PNG')
                cropped_image.save(cropped_path, format='PNG')

                # Predict with the model
                # results = model(r'D://code//data//1.28//cropped_image.png', show=False)
                results = model(cropped_path, show=False)

                # Load the original image
                # original_image = cv2.imread(r'D://code//data//1.28//cropped_image.png')
                original_image = cv2.imread(cropped_path)

                # Create a copy of the original image to modify
                modified_image = original_image.copy()
                results = model.predict(image_np, show=False)

                if len(results) > 0:
                    result = results[0]
                    if len(result.boxes) > 0:
                        # Loop through each mask found by the model and remove them from the modified image
                        for result in results:
                            masks = result.masks.cpu().data.numpy()

                            for mask in masks:
                                mask_binary = (mask > 0.5).astype(np.uint8)
                                mask_binary_resized = cv2.resize(mask_binary, (original_image.shape[1], 
                                                                            original_image.shape[0]), 
                                                                interpolation=cv2.INTER_NEAREST)

                                # Get the indices of the mask area
                                mask_indices = np.where(mask_binary_resized == 1)

                                # Set the mask area to black in the modified image
                                modified_image[mask_indices[0], mask_indices[1]] = [0, 0, 0]

                        # Save the modified image
                        final_output_path = os.path.join(output_folder, f"{os.path.splitext(image_filename)[0]}.png")
                        cv2.imwrite(final_output_path, modified_image)
    print('所有图片分割完成')



    '''
    白底图筛选
    '''
    # 定义阈值
    n = 0.4

    if not os.path.exists(folder_B):
        os.makedirs(folder_B)
    if not os.path.exists(folder_C):
        os.makedirs(folder_C)

    # 初始化列表，用于存储每张图片中白色像素点比例
    white_ratios = []

    # 遍历文件夹中的所有图片
    for image_filename in tqdm(os.listdir(output_folder), desc='Processing images'):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(output_folder, image_filename)
            image = cv2.imread(image_path)  # 读取图片
            white_pixels = np.sum(image == [255, 255, 255]) # 计算白色像素点数量
            total_pixels = image.shape[0] * image.shape[1]  # 计算总像素点数量
            white_ratio = white_pixels / total_pixels  # 计算白色像素点比例
            white_ratios.append(white_ratio)  # 将白色像素点比例添加到列表中

            if white_ratio < n:
                shutil.copy(image_path, folder_B)
            else:
                shutil.copy(image_path, folder_C)
    print('白底图分拣完成') 



    '''
    基于复杂度分类color_simple和color_mix
    '''
    import cv2
    import os
    import shutil

    # folder_path = "D://code//data//2.28//img_color"  # 文件夹路径
    folder_path = folder_B  # 文件夹路径

    # canny_low_threshold = 30
    # canny_high_threshold = 50
    # threshold_percentage = 0.012  # 阈值白色像素点占比

    # output_folder_A = "D://code//data//2.28//img_color_canny_a"  # 文件夹A的路径
    # output_folder_B = "D://code//data//2.28//img_color_canny_b"  # 文件夹B的路径

    os.makedirs(output_folder_A, exist_ok=True)
    os.makedirs(output_folder_B, exist_ok=True)


    # 如果白色像素点占比小于threshold_percentage，则将原图复制到文件夹A；
    # 如果占比大于threshold_percentage，则将原图复制到文件夹B
    #
    def calculate_white_pixel_ratio(image):
        total_pixels = image.shape[0] * image.shape[1]
        white_pixels = cv2.countNonZero(image)
        white_pixel_ratio = white_pixels / total_pixels
        return white_pixels, white_pixel_ratio

    for image_file in os.listdir(folder_path):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)     
            edges = cv2.Canny(image, canny_low_threshold, canny_high_threshold)
            white_pixels, white_pixel_ratio = calculate_white_pixel_ratio(edges)
            
            if white_pixel_ratio < threshold_percentage:
                output_path = os.path.join(output_folder_A, image_file)
                shutil.copyfile(image_path, output_path)
                print(f"Image {image_file} img_color_canny_a.")
            else:
                output_path = os.path.join(output_folder_B, image_file)
                shutil.copyfile(image_path, output_path)
                print(f"Image {image_file} img_color_canny_b.")

                
                
    import cv2
    import os
    import shutil

    folder_path = output_folder_A  # 文件夹路径
    # folder_path = "D://code//data//2.28//img_color_canny_a"

    # black_pixel_threshold = 0.85  # 黑色像素点占比阈值

    # output_folder_C = "D://code//data//2.28//img_color_canny_c"  # 文件夹A的路径
    os.makedirs(output_folder_C, exist_ok=True)

    def calculate_black_pixel_ratio(image):
        total_pixels = image.shape[0] * image.shape[1]
        black_pixels = total_pixels - cv2.countNonZero(image)
        black_pixel_ratio = black_pixels / total_pixels
        return black_pixel_ratio

    for image_file in os.listdir(folder_path):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            black_pixel_ratio = calculate_black_pixel_ratio(image)
    #         print(image_file, black_pixel_ratio)
            
            if black_pixel_ratio > black_pixel_threshold:
                output_path = os.path.join(output_folder_C, image_file)
                shutil.move(image_path, output_path)
                print(f"Image {image_file} img_color_canny_c.")
    print('完成')



    '''
    色值提取
    '''
    from PIL import Image
    import os
    from colorthief import ColorThief
    import pandas as pd
    from collections import Counter
    from tqdm import tqdm

    # 定义文件夹路径
    # folder_path = 'D://code//data//2.28//img_color_canny_a'
    folder_path = output_folder_A

    # output_file = 'D://code//data//2.28//img_color_canny_a//color_main.xlsx'

    data = []

    # 遍历文件夹中的每张图片
    for image_filename in tqdm(os.listdir(folder_path), desc='Processing images'):
        if image_filename.endswith(('.jpg', '.jpeg', '.png')):  # 仅处理特定格式的图片文件
            image_path = os.path.join(folder_path, image_filename)
            
            # 统计颜色信息
            colors = []
            img = Image.open(image_path)
            img = img.convert("RGB")
            pixels = img.load()
            for i in range(img.size[0]):  # 宽
                for j in range(img.size[1]):  # 高
                    r, g, b = img.getpixel((i, j))
                    if r > 10 or g > 10 or b > 10:  # 排除黑色像素（RGB 值大于 10）
                        colors.append((r, g, b))
            
            color_thief = ColorThief(image_path)
            palette = color_thief.get_palette(color_count=5, quality=10)
            most_common_color = Counter(colors).most_common(1)[0][0]  # 获取出现次数最多的颜色值
            
            # 拆分最常见颜色值
            main_color = ','.join(map(str, most_common_color[:3]))  # 取RGB前三个值作为主要颜色
            
            color_counter = Counter(colors)
            total_pixels = sum(color_counter.values())
            
            color_data = [{'Color': ','.join(map(str, color)), 
                        'Pixel Count': count, 
                        'Pixel Percentage': count / total_pixels * 100} 
                        for color, count in color_counter.items()]
            
            data.append({'Image Name': image_filename, 
    #                      'Colors': '--'.join([','.join(map(str, color)) for color in colors]), 
    #                      'Most Common Color': ','.join(map(str, most_common_color)),
                        'Main Color': main_color,
                        'Color Data': color_data})

    # 将数据存入DataFrame
    df = pd.DataFrame(data)

    # 将DataFrame保存为Excel文件
    df.to_excel(output_file, index=False)

    print('Color data saved to', output_file)



    '''
    图片色彩聚类
    '''
    import numpy as np
    from sklearn.cluster import KMeans
    from skimage import color
    import pandas as pd
    import matplotlib.pyplot as plt

    # 从Excel文件中读取指定列数据
    # data = pd.read_excel('D://code//data//2.28//img_color_canny_a//color_main.xlsx')  # 替换为您的Excel文件路径和工作表名称
    data = pd.read_excel(output_file)  # 替换为您的Excel文件路径和工作表名称

    selected_column = 'Main Color'  # 替换为您想要选择的列名

    #######
    # 检测指定列是否为null, 并跳过空值
    selected_column = 'Main Color'  # 替换为您想要选择的列名
    if selected_column in data.columns:
        # 从指定列中获取数据
        rgb_values = data[selected_column]
        # 过滤空值
        rgb_values = rgb_values.dropna()
        # 转换为列表
        rgb_values = rgb_values.tolist()
    else:
        print(f"指定的列'{selected_column}'不存在。")
        continue


    # rgb_values = data[selected_column].values.reshape(-1, 3)  # 假设每行包含RGB值的3列
    # rgb_values = data[selected_column]  # 假设每行包含RGB值的3列
    # rgb_values = rgb_values.tolist()

    def convert_color_strings_to_ints(color_strings):
        # 初始化一个空列表来存储转换后的颜色
        converted_colors = []
        
        # 遍历输入的字符串列表
        for color_str in color_strings:
            # 使用逗号分割字符串，并转换为整数列表
            int_color = [int(color) for color in color_str.split(',')]
            # 将整数列表添加到结果列表中
            converted_colors.append(int_color)
        
        return converted_colors

    converted_colors = convert_color_strings_to_ints(rgb_values)
    X = np.array(converted_colors)



    def distance(point1, point2):  
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def k_means(data, k, max_iter=10000):
        centers = {}  
        n_data = data.shape[0]   
        for idx, i in enumerate(random.sample(range(n_data), k)):
            centers[idx] = data[i]  

        for i in range(max_iter):  
    #         print("开始第{}次迭代".format(i+1))
            clusters = {}    
            for j in range(k):  
                clusters[j] = []
                
            for sample_idx, sample in enumerate(data):  
                distances = []  
                for c in centers:  
                    distances.append(distance(sample, centers[c])) 
                idx = np.argmin(distances)  
                clusters[idx].append(sample_idx)   # 将该样本的索引添加到对应聚类中心的列表
                
            pre_centers = centers.copy()  

            for c in clusters.keys():
                if len(clusters[c]) > 0:
                    centers[c] = np.mean(data[clusters[c]], axis=0)
    
            is_convergent = True
            for c in centers:
                if distance(pre_centers[c], centers[c]) > 1e-8:  
                    is_convergent = False
                    break
            if is_convergent == True:  
                break
        labels = np.zeros(n_data)
        for c, samples in clusters.items():
            for sample_idx in samples:
                labels[sample_idx] = c
        
        return centers, clusters, labels


    #####
    try:
        centers, clusters, labels = k_means(X, 5)

    except ValueError as e:
        print("K-Means 聚类失败，请检查输入数据。")  # 聚类失败，请检查输入数据。
        continue

    else:

        # 测试数据
        data = np.random.rand(100, 2)
        k = 5
        centers, clusters, labels = k_means(X, k)

        # 输出每个样本的分类标签
        print(centers)

        # 统计每个label出现的数量
        from collections import Counter
        label_counts = Counter(labels)
        print(label_counts)

        # 统计每个label出现的数量
        from collections import Counter
        label_counts = Counter(labels)

        # 按数量降序排序
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

        # 获取排好序的标签和对应的中心颜色
        sorted_labels, sorted_centers = zip(*[(l, centers[l]) for l, _ in sorted_labels])

        # print("Sorted Centers:")
        # for center in sorted_centers:
        #     print(center)

        import matplotlib.pyplot as plt

        # 创建DataFrame来保存聚类中心
        centers_df = pd.DataFrame(centers, columns=['x1', 'x2', 'x3'])
        # 添加计数数量
        centers_df['Count'] = [label_counts[label] for label in range(k)]

        # 创建DataFrame来保存每个标签的计数
        label_counts_df = pd.DataFrame.from_dict(label_counts, orient='index', columns=['Count'])
        label_counts_df.index.name = 'Label'  # 设置索引名称

        # 创建DataFrame来保存排序后的标签和中心
        sorted_labels_df = pd.DataFrame({'Label': sorted_labels, 'Center': sorted_centers})
        # 添加计数数量
        sorted_labels_df['Count'] = [label_counts[label] for label in sorted_labels]

        # 将DataFrame保存到Excel文件的不同sheet中
        # with pd.ExcelWriter(base_path + 'output.xlsx') as writer:
        with pd.ExcelWriter(folder_path + 'output.xlsx') as writer:

        #     centers_df.to_excel(writer, sheet_name='Centers', index=False)
        #     label_counts_df.to_excel(writer, sheet_name='LabelCounts')
            sorted_labels_df.to_excel(writer, sheet_name='SortedLabels', index=False)

        print("聚类结果已保存到'output.xlsx'文件中。")


    # 原始的元组列表
    colors_tuple = (sorted_centers)

    # 转换为字典
    colors_dict = {i: color for i, color in enumerate(colors_tuple)}

    # 对字典的键进行排序并逆序
    sorted_keys = sorted(colors_dict.keys(), reverse=True)
    colors_sorted = {key: colors_dict[key] for key in sorted_keys}

    # 绘制色块图
    def plot_colors(colors, save_path=None):
        fig, ax = plt.subplots(figsize=(4, 8))  # 设置画板大小

        # 绘制每个色块并填充色值
        for i, (key, rgb) in enumerate(colors.items()):
            ax.barh(i, 1, color=[c/255 for c in rgb], edgecolor='black')
            ax.text(1.2, i, f"RGB: {rgb}", ha='left', va='center', color='black', fontsize=10)

        # 隐藏坐标轴
        ax.axis('off')

        # 保存图像
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        # # 显示图像
        # plt.show()

    # 示例的 RGB 色值字典
    colors = colors_sorted  # 使用逆序后的字典

    # 绘制色块图并保存为图片
    save_path = folder_path + 'color_block_chart.png'
    plot_colors(colors, save_path)

