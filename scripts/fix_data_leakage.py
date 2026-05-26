#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复数据泄漏 - 移除重复图片
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"


def compute_md5(filepath):
    """计算文件 MD5 哈希"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_image_hashes(data):
    """获取所有图片的 MD5 哈希"""
    hashes = defaultdict(list)
    for record in data:
        for img_path in record.get('images', []):
            full_path = ROOT_DIR / img_path
            if full_path.exists():
                md5 = compute_md5(full_path)
                hashes[md5].append(record['id'])
    return hashes


def main():
    print("="*60)
    print("Fixing Data Leakage - Removing Duplicate Images")
    print("="*60)
    
    # 加载数据
    print("\nLoading data files...")
    with open(DATA_DIR / "dataset_train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(DATA_DIR / "dataset_val.json", 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open(DATA_DIR / "dataset_test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"  Train: {len(train_data)} records")
    print(f"  Val: {len(val_data)} records")
    print(f"  Test: {len(test_data)} records")
    
    # 获取图片哈希
    print("\nComputing image MD5 hashes...")
    train_hashes = get_image_hashes(train_data)
    test_hashes = get_image_hashes(test_data)
    
    # 找出重复的哈希
    common_hashes = set(train_hashes.keys()) & set(test_hashes.keys())
    
    if not common_hashes:
        print("\n[OK] No duplicate images found!")
        return
    
    print(f"\n[WARNING] Found {len(common_hashes)} duplicate MD5 hashes between train and test!")
    
    # 收集需要从 test 中移除的 ID
    ids_to_remove_from_test = set()
    for h in common_hashes:
        # 从 test 中移除这些记录
        for record_id in test_hashes[h]:
            ids_to_remove_from_test.add(record_id)
            print(f"  Will remove from test: {record_id} (MD5: {h[:16]}...)")
    
    # 过滤测试数据
    test_data_clean = [r for r in test_data if r['id'] not in ids_to_remove_from_test]
    
    print(f"\nOriginal test size: {len(test_data)}")
    print(f"Cleaned test size: {len(test_data_clean)}")
    print(f"Removed: {len(test_data) - len(test_data_clean)} records")
    
    # 保存清洗后的数据
    backup_dir = DATA_DIR / "backup_before_fix"
    backup_dir.mkdir(exist_ok=True)
    
    # 备份原始文件
    import shutil
    shutil.copy(DATA_DIR / "dataset_test.json", backup_dir / "dataset_test.json.bak")
    print(f"\nBackup saved to: {backup_dir}")
    
    # 保存清洗后的数据
    with open(DATA_DIR / "dataset_test.json", 'w', encoding='utf-8') as f:
        json.dump(test_data_clean, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Cleaned test data saved!")
    
    # 验证
    print("\nVerifying fix...")
    test_hashes_new = get_image_hashes(test_data_clean)
    common_hashes_new = set(train_hashes.keys()) & set(test_hashes_new.keys())
    
    if not common_hashes_new:
        print("[OK] No duplicate images between train and test after fix!")
    else:
        print(f"[ERROR] Still have {len(common_hashes_new)} duplicates!")
    
    print("\n" + "="*60)
    print("Data leakage fix complete!")
    print("="*60)


if __name__ == "__main__":
    main()
