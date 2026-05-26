from bing_image_downloader import downloader
import os
import shutil
import hashlib

def get_file_hash(filepath):
    """计算文件的 MD5 哈希值，用于精确的图片去重"""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception:
        return None

def build_massive_dataset():
    img_dir = "sample_images"
    os.makedirs(img_dir, exist_ok=True)
    
    # 1. 建立现有图片的哈希库，防止重复下载相同的图
    print("正在扫描本地已有图片，建立去重指纹库...")
    existing_hashes = set()
    category_max_index = {}
    
    for f in os.listdir(img_dir):
        path = os.path.join(img_dir, f)
        if os.path.isfile(path):
            file_hash = get_file_hash(path)
            if file_hash:
                existing_hashes.add(file_hash)
            
            # 记录类别的最大标号，实现真正的增量更新
            if "_" in f:
                parts = f.rsplit("_", 1)
                cat = parts[0]
                try:
                    num = int(parts[1].split(".")[0])
                    category_max_index[cat] = max(category_max_index.get(cat, 0), num)
                except ValueError:
                    pass

    print(f"本地已存在 {len(existing_hashes)} 张不重复的图片。准备进行增量下载！\n")

    # 为了防止单一关键词搜不到足够的图，我们将每个灾害细分为3个不同的长尾关键词
    queries = {
        "flood_洪涝": [
            "severe flood disaster damage",
            "urban flooding submerged vehicles",
            "river flood aerial view aftermath",
            "flash flood destroying houses",
            "flood victims rescue boat",
            "flooded streets heavy rain damage"
        ],
        "earthquake_地震": [
            "earthquake building collapse disaster scene",
            "earthquake rescue rubble destruction",
            "seismic damage road crack earthquake",
            "magnitude earthquake aftermath ruins",
            "collapsed bridge earthquake",
            "earthquake survivor debris rescue"
        ],
        "fire_火灾": [
            "wildfire forest fire destruction",
            "house burned down fire disaster",
            "firefighters battling massive wildfire",
            "building engulfed in flames",
            "bushfire aerial view damage",
            "scorched earth wildfire aftermath"
        ],
        "landslide_滑坡": [
            "landslide mudslide road damage",
            "mountain landslide burying houses",
            "heavy rain mudslide disaster",
            "rockfall blocking highway",
            "earth slip destroying village",
            "landslide aerial view rescue"
        ],
        "typhoon_台风": [
            "typhoon hurricane storm destruction aftermath",
            "cyclone damage uprooted trees",
            "hurricane wind damage destroyed roofs",
            "storm surge flooding coastal",
            "tornado path of destruction",
            "severe storm debris street"
        ]
    }
    
    total_new_downloaded = 0
    
    for category, keywords in queries.items():
        print(f"\n>>> 正在处理大类: [{category}]")
        # 获取当前类别的起始编号
        current_idx = category_max_index.get(category, 0)
        
        for query in keywords:
            print(f"  -> 正在执行细分关键词: '{query}'")
            try:
                downloader.download(query, limit=200, output_dir='temp_downloads', adult_filter_off=False, force_replace=False, timeout=15, verbose=False)
                
                downloaded_folder = os.path.join('temp_downloads', query)
                if os.path.exists(downloaded_folder):
                    files = os.listdir(downloaded_folder)
                    success_count = 0
                    for filename in files:
                        ext = os.path.splitext(filename)[1]
                        if ext.lower() not in ['.jpg', '.jpeg', '.png']:
                            continue
                        
                        src = os.path.join(downloaded_folder, filename)
                        file_hash = get_file_hash(src)
                        
                        # 【核心逻辑】如果哈希值已存在，说明图片重复，直接丢弃
                        if file_hash in existing_hashes:
                            continue
                            
                        # 如果是不重复的新图
                        current_idx += 1
                        new_name = f"{category}_{current_idx}{ext}"
                        dst = os.path.join(img_dir, new_name)
                        shutil.copy(src, dst)
                        
                        existing_hashes.add(file_hash)
                        success_count += 1
                        total_new_downloaded += 1
                        
                    print(f"     [+] 关键词 '{query}' 增量新增了 {success_count} 张不重复图片。")
            except Exception as e:
                print(f"     [x] 抓取 '{query}' 时出现异常: {e}")
            
    # 清理临时下载文件夹
    if os.path.exists('temp_downloads'):
        shutil.rmtree('temp_downloads')
                
    print("\n==================================================")
    print(f"增量爬取任务执行完毕！本次共新增入库 {total_new_downloaded} 张全新的灾害图片。")

if __name__ == "__main__":
    build_massive_dataset()
