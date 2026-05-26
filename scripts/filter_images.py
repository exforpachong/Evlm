import os
from PIL import Image
import warnings

# 忽略 PIL 的一些非致命警告
warnings.filterwarnings('ignore')

def filter_invalid_images():
    img_dir = "sample_images"
    if not os.path.exists(img_dir):
        print(f"目录 {img_dir} 不存在。")
        return
        
    invalid_count = 0
    total_count = 0
    print("==================================================")
    print("开始深度扫描并清理无效/损坏的图片...")
    print("==================================================")
    
    for filename in os.listdir(img_dir):
        filepath = os.path.join(img_dir, filename)
        if not os.path.isfile(filepath):
            continue
            
        total_count += 1
        
        # 1. 过滤：检查文件大小 (小于 10KB 往往是网页图标或无效空文件)
        if os.path.getsize(filepath) < 10 * 1024:
            try:
                os.remove(filepath)
                invalid_count += 1
                print(f"  [移除] {filename} (文件过小，可能是占位图)")
            except Exception as e:
                pass
            continue
            
        # 2. 过滤：检查图片文件是否损坏，或本质是不是HTML网页等
        try:
            with Image.open(filepath) as img:
                img.load() # 尝试读取全部像素数据，严苛验证图片完整性
                
                # 3. 过滤：剔除非常极端的长宽比或极小分辨率的图片 (比如宽或高小于100像素)
                if img.width < 100 or img.height < 100:
                    raise ValueError("图片分辨率过小")
                    
        except Exception as e:
            try:
                os.remove(filepath)
                invalid_count += 1
                print(f"  [移除] {filename} (图片已损坏或格式错误: {e})")
            except Exception:
                pass
                
    valid_count = total_count - invalid_count
    print("\n==================================================")
    print(f"数据集自动清洗完成！")
    print(f"报告：共检查了 {total_count} 个文件。")
    print(f"移除了 {invalid_count} 张无效、损坏或非图像文件。")
    print(f"最终保留的高质量灾害图片总数: {valid_count} 张。")
    print("==================================================")

if __name__ == "__main__":
    filter_invalid_images()
