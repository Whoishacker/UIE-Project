import os
from PIL import Image


def convert_images_to_jpg(folder_path):
    """
    将指定文件夹内所有图片文件转换为jpg格式，并删除原文件

    参数:
        folder_path: 图片所在文件夹路径
    """
    # 支持的图片文件扩展名（不包括jpg/jpeg，避免重复处理）
    image_extensions = ('.png', '.gif', '.bmp', '.tiff', '.webp')

    # 遍历文件夹内所有文件
    for filename in os.listdir(folder_path):
        # 获取文件完整路径
        file_path = os.path.join(folder_path, filename)

        # 跳过文件夹，只处理文件
        if os.path.isdir(file_path):
            continue

        # 检查文件是否为需要转换的图片格式
        file_lower = filename.lower()
        if file_lower.endswith(image_extensions):
            try:
                # 打开图片
                with Image.open(file_path) as img:
                    # 创建新的文件名（去掉原扩展名，加上.jpg）
                    name_without_ext = os.path.splitext(filename)[0]
                    new_filename = f"{name_without_ext}.jpg"
                    new_file_path = os.path.join(folder_path, new_filename)

                    # 如果目标文件已存在，先删除原文件
                    if os.path.exists(new_file_path):
                        print(f"目标文件已存在，删除原文件: {filename}")
                        os.remove(file_path)
                        continue

                    # 转换为RGB模式（处理透明通道等问题）
                    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                        # 透明背景替换为白色
                        background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                        background.paste(img, img.split()[-1])
                        img = background
                    elif img.mode == 'P':
                        img = img.convert('RGB')

                    # 保存为jpg格式
                    img.save(new_file_path, 'JPEG', quality=95)
                    print(f"转换成功并删除原文件: {filename} -> {new_filename}")

                    # 删除原文件
                    os.remove(file_path)

            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")


if __name__ == "__main__":
    # 目标文件夹路径（修改为你的文件夹路径）
    target_folder = "testA"  # 可以是绝对路径或相对路径

    # 检查文件夹是否存在
    if not os.path.exists(target_folder):
        print(f"错误: 文件夹 '{target_folder}' 不存在")
    elif not os.path.isdir(target_folder):
        print(f"错误: '{target_folder}' 不是一个文件夹")
    else:
        print(f"开始转换文件夹 '{target_folder}' 中的图片...")
        convert_images_to_jpg(target_folder)
        print("转换完成！只保留JPG文件")