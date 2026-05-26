#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据审计脚本 - 建立 Benchmark 契约
功能：
1. 数据泄漏检查（MD5 哈希）
2. 数据统计输出
3. 生成版本标记文件
"""

import json
import hashlib
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import re

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
SAMPLE_IMAGES_DIR = ROOT_DIR / "sample_images"
OUTPUT_DIR = ROOT_DIR / "release_package"


def load_json(filepath):
    """加载 JSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_md5(filepath):
    """计算文件 MD5 哈希"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def parse_assistant_response(response_str):
    """解析 assistant 的 JSON 响应"""
    try:
        # 尝试直接解析
        return json.loads(response_str)
    except json.JSONDecodeError:
        # 尝试提取 JSON 部分
        json_match = re.search(r'\{[^{}]*\}', response_str, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
    return None


def extract_labels_from_record(record):
    """从记录中提取标签"""
    for conv in record.get('conversations', []):
        if conv.get('from') == 'assistant':
            parsed = parse_assistant_response(conv.get('value', ''))
            if parsed:
                return {
                    'disaster_type': parsed.get('disaster_type'),
                    'damage_count': parsed.get('damage_count'),
                    'object_relations': parsed.get('object_relations'),
                    'report': parsed.get('report')
                }
    return None


def check_data_leakage(train_data, test_data, val_data=None):
    """检查数据泄漏"""
    print("\n" + "="*60)
    print("[Data Leakage Check]")
    print("="*60)
    
    # 收集所有图片 ID
    train_ids = set(r['id'] for r in train_data)
    test_ids = set(r['id'] for r in test_data)
    
    # 检查 ID 重叠
    id_overlap = train_ids & test_ids
    if id_overlap:
        print(f"[WARNING] Found {len(id_overlap)} IDs in both train and test!")
        print(f"   Overlap ID samples: {list(id_overlap)[:5]}")
    else:
        print("[OK] No ID overlap")
    
    # 检查图片文件 MD5 重叠
    print("\n检查图片文件 MD5...")
    
    def get_image_hashes(data, split_name):
        hashes = {}
        for record in data:
            for img_path in record.get('images', []):
                full_path = ROOT_DIR / img_path
                if full_path.exists():
                    md5 = compute_md5(full_path)
                    if md5 not in hashes:
                        hashes[md5] = []
                    hashes[md5].append((record['id'], split_name, img_path))
        return hashes
    
    train_hashes = get_image_hashes(train_data, 'train')
    test_hashes = get_image_hashes(test_data, 'test')
    
    # 查找 MD5 重叠
    common_hashes = set(train_hashes.keys()) & set(test_hashes.keys())
    if common_hashes:
        print(f"\n[WARNING] Found {len(common_hashes)} duplicate images in train and test!")
        for h in list(common_hashes)[:3]:
            print(f"   MD5: {h[:16]}...")
            for item in train_hashes[h] + test_hashes[h]:
                print(f"      - {item}")
    else:
        print("[OK] No image MD5 overlap")
    
    # 检查同源（按 ID 前缀）
    print("\n检查同源数据...")
    train_prefixes = Counter(id.split('_')[0] + '_' + id.split('_')[1] if '_' in id else id for id in train_ids)
    test_prefixes = Counter(id.split('_')[0] + '_' + id.split('_')[1] if '_' in id else id for id in test_ids)
    
    print(f"   Train 数据源分布: {dict(train_prefixes.most_common(10))}")
    print(f"   Test 数据源分布: {dict(test_prefixes.most_common(10))}")
    
    leakage_report = {
        'id_overlap_count': len(id_overlap),
        'image_md5_overlap_count': len(common_hashes),
        'train_source_distribution': dict(train_prefixes),
        'test_source_distribution': dict(test_prefixes)
    }
    
    return leakage_report


def compute_statistics(data, split_name):
    """计算数据统计"""
    stats = {
        'total_count': len(data),
        'disaster_types': Counter(),
        'severities': Counter(),
        'damage_counts': defaultdict(int),
        'missing_fields': defaultdict(int),
        'image_resolutions': [],
        'parse_errors': 0
    }
    
    for record in data:
        labels = extract_labels_from_record(record)
        
        if labels is None:
            stats['parse_errors'] += 1
            continue
        
        # 灾害类型
        dt = labels.get('disaster_type')
        if dt:
            # 标准化灾害类型
            dt_normalized = dt.lower().strip()
            if 'flood' in dt_normalized or '洪水' in dt_normalized or '洪涝' in dt_normalized:
                stats['disaster_types']['flood'] += 1
            elif 'earthquake' in dt_normalized or '地震' in dt_normalized:
                stats['disaster_types']['earthquake'] += 1
            elif 'fire' in dt_normalized or '火灾' in dt_normalized or 'wildfire' in dt_normalized:
                stats['disaster_types']['fire'] += 1
            elif 'landslide' in dt_normalized or '滑坡' in dt_normalized or '泥石流' in dt_normalized:
                stats['disaster_types']['landslide'] += 1
            elif 'typhoon' in dt_normalized or '台风' in dt_normalized:
                stats['disaster_types']['typhoon'] += 1
            else:
                stats['disaster_types'][dt] += 1
        else:
            stats['missing_fields']['disaster_type'] += 1
        
        # 损毁数量
        dc = labels.get('damage_count')
        if dc is not None:
            try:
                # 尝试转换为数字
                if isinstance(dc, str):
                    # 提取数字
                    nums = re.findall(r'\d+', dc)
                    if nums:
                        dc_int = int(nums[0])
                        stats['damage_counts'][dc_int] += 1
                    else:
                        stats['damage_counts']['unknown'] += 1
                else:
                    stats['damage_counts'][int(dc)] += 1
            except:
                stats['damage_counts']['unknown'] += 1
        else:
            stats['missing_fields']['damage_count'] += 1
        
        # 检查其他字段
        for field in ['object_relations', 'report']:
            if not labels.get(field):
                stats['missing_fields'][field] += 1
    
    return stats


def print_statistics(stats, split_name):
    """打印统计信息"""
    print(f"\n{'='*60}")
    print(f"[{split_name} Data Statistics]")
    print("="*60)
    
    print(f"\n总样本数: {stats['total_count']}")
    print(f"解析错误: {stats['parse_errors']}")
    
    print(f"\n灾害类型分布:")
    for dt, count in stats['disaster_types'].most_common():
        pct = count / stats['total_count'] * 100
        print(f"  {dt}: {count} ({pct:.1f}%)")
    
    print(f"\n损毁数量分布 (前10):")
    sorted_dc = sorted([(k, v) for k, v in stats['damage_counts'].items() if isinstance(k, int)], key=lambda x: x[0])
    for dc, count in sorted_dc[:10]:
        print(f"  {dc}: {count}")
    if 'unknown' in stats['damage_counts']:
        print(f"  unknown: {stats['damage_counts']['unknown']}")
    
    print(f"\n字段缺失统计:")
    for field, count in stats['missing_fields'].items():
        pct = count / stats['total_count'] * 100
        print(f"  {field}: {count} ({pct:.1f}%)")


def generate_metadata(train_stats, val_stats, test_stats, leakage_report):
    """生成版本元数据"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    metadata = {
        "dataset_name": "Multi_Disaster_Visual_Language_Benchmark",
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "description": "面向应急管理的多灾种视觉-语言数据集",
        "splits": {
            "train": {
                "count": train_stats['total_count'],
                "file": "data/dataset_train.json"
            },
            "val": {
                "count": val_stats['total_count'],
                "file": "data/dataset_val.json"
            },
            "test": {
                "count": test_stats['total_count'],
                "file": "data/dataset_test.json"
            }
        },
        "disaster_types": {
            "train": dict(train_stats['disaster_types']),
            "val": dict(val_stats['disaster_types']),
            "test": dict(test_stats['disaster_types'])
        },
        "tasks": [
            "disaster_type_classification",
            "damage_count_estimation", 
            "object_relation_reasoning",
            "disaster_report_generation"
        ],
        "data_leakage_check": {
            "id_overlap": leakage_report['id_overlap_count'],
            "image_md5_overlap": leakage_report['image_md5_overlap_count'],
            "status": "PASS" if leakage_report['id_overlap_count'] == 0 and leakage_report['image_md5_overlap_count'] == 0 else "WARNING"
        },
        "annotation_process": "GLM-4.6V 自动标注 + 人工复核",
        "license": "待定",
        "contact": ""
    }
    
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Metadata saved: {metadata_path}")
    return metadata


def main():
    print("="*60)
    print("Multi-Disaster Benchmark Audit Tool")
    print("="*60)
    
    # 加载数据
    print("\n加载数据文件...")
    train_data = load_json(DATA_DIR / "dataset_train.json")
    val_data = load_json(DATA_DIR / "dataset_val.json")
    test_data = load_json(DATA_DIR / "dataset_test.json")
    
    print(f"  Train: {len(train_data)} 条")
    print(f"  Val: {len(val_data)} 条")
    print(f"  Test: {len(test_data)} 条")
    
    # 检查数据泄漏
    leakage_report = check_data_leakage(train_data, test_data, val_data)
    
    # 计算统计
    print("\n计算统计数据...")
    train_stats = compute_statistics(train_data, 'train')
    val_stats = compute_statistics(val_data, 'val')
    test_stats = compute_statistics(test_data, 'test')
    
    # 打印统计
    print_statistics(train_stats, 'Train')
    print_statistics(val_stats, 'Val')
    print_statistics(test_stats, 'Test')
    
    # 生成元数据
    metadata = generate_metadata(train_stats, val_stats, test_stats, leakage_report)
    
    # 保存详细统计
    detailed_stats = {
        "train": {
            "total": train_stats['total_count'],
            "disaster_types": dict(train_stats['disaster_types']),
            "damage_counts": {str(k): v for k, v in train_stats['damage_counts'].items()},
            "missing_fields": dict(train_stats['missing_fields']),
            "parse_errors": train_stats['parse_errors']
        },
        "val": {
            "total": val_stats['total_count'],
            "disaster_types": dict(val_stats['disaster_types']),
            "damage_counts": {str(k): v for k, v in val_stats['damage_counts'].items()},
            "missing_fields": dict(val_stats['missing_fields']),
            "parse_errors": val_stats['parse_errors']
        },
        "test": {
            "total": test_stats['total_count'],
            "disaster_types": dict(test_stats['disaster_types']),
            "damage_counts": {str(k): v for k, v in test_stats['damage_counts'].items()},
            "missing_fields": dict(test_stats['missing_fields']),
            "parse_errors": test_stats['parse_errors']
        },
        "leakage_report": leakage_report
    }
    
    stats_path = DATA_DIR / "data_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_stats, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Detailed statistics saved: {stats_path}")
    
    # 最终报告
    print("\n" + "="*60)
    print("[Audit Summary]")
    print("="*60)
    
    total_samples = len(train_data) + len(val_data) + len(test_data)
    print(f"\n总样本数: {total_samples}")
    print(f"数据划分: Train {len(train_data)} / Val {len(val_data)} / Test {len(test_data)}")
    
    if leakage_report['id_overlap_count'] == 0 and leakage_report['image_md5_overlap_count'] == 0:
        print("\n[OK] Data leakage check: PASS")
    else:
        print(f"\n[WARNING] Data leakage check: Issues found")
        print(f"   ID overlap: {leakage_report['id_overlap_count']}")
        print(f"   Image MD5 overlap: {leakage_report['image_md5_overlap_count']}")
    
    print("\n[OK] Benchmark contract established!")
    print(f"   Data version: multi_disaster_visual_language_benchmark_v1")
    print(f"   Metadata file: release_package/metadata.json")


if __name__ == "__main__":
    main()
