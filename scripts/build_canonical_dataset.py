#!/usr/bin/env python3
"""
Build canonical dataset with normalized disaster_type labels.
Maps all variants to 6 canonical categories:
- flood
- earthquake  
- fire
- landslide
- windstorm_or_typhoon
- other
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# Canonical label mapping rules
CANONICAL_LABELS = {
    "flood": ["flood", "洪水", "洪涝", "暴雨", "城市内涝", "山洪", "内涝", "内涝（暴雨引发）", "风暴潮", "风暴潮（或海啸）引发的沿海灾害"],
    "earthquake": ["earthquake", "地震", "地裂", "地裂（地质灾害）", "地质灾害（地裂）"],
    "fire": ["fire", "火灾", "野火", "山火", "森林火灾", "森林灾害（林木损毁）"],
    "landslide": ["landslide", "滑坡", "泥石流", "mudslide", "山体崩塌", "塌方", "塌方（路基塌陷）", "道路坍塌", "道路坍塌（地质灾害）", "地质灾害（道路坍塌）", "山体崩塌", "岩石崩塌", "地面塌陷", "地面塌陷（道路损毁）", "地面塌陷（地质灾害）", "坍塌", "坍塌（建筑/山体坍塌）"],
    "windstorm_or_typhoon": [
        "typhoon", "台风", "风灾", "飓风", "龙卷风", "强风", "风暴", "windstorm",
        "tornado", "热带气旋", 
        # Chinese variants
        "风灾（风暴灾害）", "强风灾害", "飓风（强风灾害）", "风暴灾害（强风导致树木倒伏）",
        "龙卷风（强风灾害）", "飓风（沿海强风灾害）", "飓风（风暴灾害）", "风灾（强风）",
        "风暴（飓风）灾害", "风灾（极端天气引发的设施损毁）", "强风灾害（树木倒塌）",
        "风灾（飓风）", "飓风（热带气旋灾害）", "风暴（强风）灾害", "风暴（风灾）",
        "飓风（热带气旋）", "飓风（或龙卷风）", "风暴灾害（如飓风）", "风灾（强风导致树木倒塌）",
        "海岸侵蚀（风暴潮引发的地基坍塌）", "强风灾害（疑似飓风或龙卷风）",
        "强风灾害（如飓风/龙卷风）", "强风（风暴）", "风暴（强风）", "强风（风暴）灾害",
        "风灾（飓风/龙卷风）", "强风灾害（风暴）", "风灾（强风/风暴）", "风灾（或飓风）",
        "风灾（建筑结构受损）", "风暴（或飓风）", "飓风（热带气旋引发的沿海灾害）",
        "风灾（强风导致的树木倒伏）", "风灾（树木倒塌致灾）", "风灾（强风/飓风）",
        "风灾（树木倒塌）", "风灾（风暴/龙卷风）", "风灾（强风灾害）",
        "沙尘暴（风沙灾害）", "风灾（强风致树木倒塌）", "热带气旋（飓风）",
        "强风（飓风）灾害", "风暴（强风灾害）", "雷暴（强对流天气）",
        "风灾（强风导致屋顶瓦片损坏）", "风灾（强风/龙卷风）",
        "极端风暴（或超级龙卷风/强对流灾害）", "风灾（如飓风/龙卷风）",
        "风灾（倒树灾害）", "风灾（如飓风）", "风灾（强风导致树木倒伏）",
        "强风灾害（风暴灾害）", "飓风（风灾）", "风灾（强风致树木倒塌）",
        "飓风（强风）灾害", "龙卷风（强风）"
    ],
    "other": [
        "other", "unknown", "其他",
        # Non-core disaster types
        "火山喷发", "火山喷发（火山灰沉降）", "volcanic eruption", "海啸",
        "建筑坍塌", "建筑物坍塌", "爆炸", "爆炸灾害", "爆炸/武装冲突", "爆炸/建筑坍塌灾害",
        "爆炸/空袭灾害", "爆炸引发的建筑坍塌", "爆炸/冲击灾害",
        "建筑损毁（结构损坏）", "干旱", "海岸侵蚀灾害", "环境污染（垃圾堆积）",
        "垃圾堆积灾害", "固体废弃物堆积", "道路损毁", "无", "无明显灾害"
    ]
}

# Build reverse mapping for fast lookup
LABEL_MAPPING = {}
for canonical, variants in CANONICAL_LABELS.items():
    for variant in variants:
        LABEL_MAPPING[variant.lower().strip()] = canonical

def normalize_label(raw_label: str) -> str:
    """Map raw label to canonical label."""
    if not raw_label:
        return "other"
    
    # Normalize: lowercase, strip whitespace
    normalized = raw_label.lower().strip()
    
    # Direct lookup
    if normalized in LABEL_MAPPING:
        return LABEL_MAPPING[normalized]
    
    # Fuzzy matching for variants not in the mapping
    # Check for key patterns
    if any(kw in normalized for kw in ["洪水", "洪涝", "内涝", "flood"]):
        return "flood"
    if any(kw in normalized for kw in ["地震", "earthquake"]):
        return "earthquake"
    if any(kw in normalized for kw in ["火", "fire", "野火", "山火"]):
        return "fire"
    if any(kw in normalized for kw in ["滑坡", "泥石流", "landslide", "塌方", "崩塌", "坍塌"]):
        return "landslide"
    if any(kw in normalized for kw in ["台风", "typhoon", "风", "飓风", "龙卷风", "storm", "tornado"]):
        return "windstorm_or_typhoon"
    
    # Default to other
    return "other"

def process_dataset(input_path: str, output_path: str) -> dict:
    """Process a dataset file and map labels to canonical form."""
    print(f"Processing: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = defaultdict(lambda: defaultdict(int))
    label_mapping_report = []
    
    for item in data:
        stats['total']['count'] += 1
        
        # Get the assistant response
        conversations = item.get('conversations', [])
        assistant_msg = None
        for conv in conversations:
            if conv.get('from') == 'assistant':
                assistant_msg = conv.get('value', '')
                break
        
        if not assistant_msg:
            stats['errors']['no_assistant_msg'] += 1
            continue
        
        # Parse JSON from assistant message
        try:
            # Extract JSON from the message (handle leading/trailing whitespace)
            json_str = assistant_msg.strip()
            # Find JSON object
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}')
            if start_idx == -1 or end_idx == -1:
                stats['errors']['no_json_found'] += 1
                continue
            
            json_content = json_str[start_idx:end_idx+1]
            parsed = json.loads(json_content)
            raw_disaster_type = parsed.get('disaster_type', '')
            
            if not raw_disaster_type:
                stats['errors']['empty_disaster_type'] += 1
                continue
            
            # Map to canonical label
            canonical = normalize_label(raw_disaster_type)
            
            # Update the message
            parsed['disaster_type'] = canonical
            parsed['disaster_type_original'] = raw_disaster_type  # Keep original for reference
            
            # Update the conversation
            for conv in conversations:
                if conv.get('from') == 'assistant':
                    conv['value'] = json.dumps(parsed, ensure_ascii=False, indent=2)
            
            # Record mapping
            stats['disaster_type'][canonical] += 1
            stats['processed']['count'] += 1
            if raw_disaster_type != canonical:
                label_mapping_report.append({
                    'original': raw_disaster_type,
                    'canonical': canonical,
                    'id': item.get('id', 'unknown')
                })
        except json.JSONDecodeError as e:
            stats['errors']['parse_error'] += 1
        except Exception as e:
            stats['errors']['other_error'] += 1
    
    # Save processed data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"  Saved to: {output_path}")
    print(f"  Total items: {len(data)}")
    print(f"  Label distribution: {dict(stats['disaster_type'])}")
    
    return {
        'stats': dict(stats),
        'mapping_report': label_mapping_report
    }

def main():
    base_path = Path(__file__).parent.parent / 'data'
    
    # Process all splits
    splits = ['train', 'val', 'test']
    all_stats = {}
    all_mappings = []
    
    for split in splits:
        input_file = base_path / f'dataset_{split}.json'
        output_file = base_path / f'dataset_{split}_canonical.json'
        
        if input_file.exists():
            result = process_dataset(str(input_file), str(output_file))
            all_stats[split] = result['stats']
            all_mappings.extend(result['mapping_report'])
        else:
            print(f"Warning: {input_file} not found")
    
    # Save mapping report
    mapping_report_path = base_path / 'canonical_label_mapping_report.csv'
    with open(mapping_report_path, 'w', encoding='utf-8') as f:
        f.write("id,original,canonical\n")
        for item in all_mappings:
            f.write(f"{item['id']},{item['original']},{item['canonical']}\n")
    
    print(f"\nMapping report saved to: {mapping_report_path}")
    
    # Save statistics
    stats_path = base_path / 'canonical_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    
    print(f"Statistics saved to: {stats_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CANONICAL LABEL DISTRIBUTION SUMMARY")
    print("="*60)
    
    canonical_labels = ["flood", "earthquake", "fire", "landslide", "windstorm_or_typhoon", "other"]
    
    print(f"\n{'Label':<25} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-" * 50)
    
    for label in canonical_labels:
        train_count = all_stats.get('train', {}).get('disaster_type', {}).get(label, 0)
        val_count = all_stats.get('val', {}).get('disaster_type', {}).get(label, 0)
        test_count = all_stats.get('test', {}).get('disaster_type', {}).get(label, 0)
        print(f"{label:<25} {train_count:>8} {val_count:>8} {test_count:>8}")
    
    print("\nNote: windstorm_or_typhoon combines typhoon, 风灾, 飓风, 龙卷风, etc.")
    print("Note: 'other' includes volcanic eruption, tsunami, building collapse, explosion, etc.")

if __name__ == '__main__':
    main()
