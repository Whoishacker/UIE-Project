import os
from PIL import Image
from tqdm import tqdm  # 进度条显示（可选，提升体验）


def resize_images_to_480x480_overwrite(input_dir):
    """
    批量将文件夹内所有图片调整为 480×480 并直接覆盖原图

    参数:
        input_dir: 输入图片文件夹路径（会递归处理子文件夹）
    """
    # 1. 校验输入目录
    if not os.path.exists(input_dir):
        print(f"错误：输入文件夹不存在 -> {input_dir}")
        return

    # 2. 支持的图片格式（常见格式全覆盖）
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.ico')

    # 3. 查找所有图片文件（包括子文件夹）
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("未找到支持的图片文件！")
        return

    # 4. 批量调整尺寸并覆盖原图
    print(f"找到 {len(image_files)} 张图片，开始调整尺寸（直接覆盖原图）...")
    for img_path in tqdm(image_files, desc="处理进度"):
        try:
            # 读取图片并高质量缩放为 480×480
            with Image.open(img_path) as img:
                # LANCZOS 算法：缩放后图片清晰度最高（支持 Python 3.9+）
                img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)

                # 直接覆盖原图（JPG 保留 95% 质量，PNG 保留透明通道）
                img_resized.save(img_path, quality=95)

        except Exception as e:
            print(f"\n警告：处理文件 {img_path} 失败 -> {str(e)}")

    print(f"\n处理完成！所有图片已调整为 480×480 并覆盖原图。")


if __name__ == "__main__":
    # -------------------------- 仅需修改这里 --------------------------
    INPUT_FOLDER = r"E:\\Underwater-project\\UGANv0_1\\data\\trainA"  # 替换为你的图片文件夹路径
    # -------------------------------------------------------------------

    # 运行程序（直接覆盖，无备份！）
    resize_images_to_480x480_overwrite(input_dir=INPUT_FOLDER)