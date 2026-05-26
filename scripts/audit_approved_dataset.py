#!/usr/bin/env python3
"""
Audit approved_dataset.jsonl for quality issues.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

DATA_FILE = Path(r"C:\Users\34791\Desktop\数据集\data\approved_dataset.jsonl")
CANONICAL_LABELS = ["flood", "earthquake", "fire", "landslide", "windstorm_or_typhoon", "other"]

# Label mapping for normalization check
LABEL_VARIANTS = {
    "earthquake": ["earthquake", "地震", "地震（建筑坍塌）"],
    "flood": ["flood", "洪水", "洪涝", "水灾"],
    "fire": ["fire", "火灾", "林火", "山火"],
    "landslide": ["landslide", "滑坡", "泥石流", "塌方"],
    "windstorm_or_typhoon": ["windstorm_or_typhoon", "台风", "风灾", "飓风", "typhoon", "windstorm"],
    "other": ["other", "其他", "unknown"],
}

def parse_json_output(text):
    if not text:
        return None
    text = text.strip()
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx == -1 or end_idx == -1:
        return None
    json_str = text[start_idx:end_idx+1]
    try:
        return json.loads(json_str)
    except:
        return None

def main():
    issues = {
        "empty_output": [],
        "truncated_json": [],
        "invalid_json": [],
        "non_canonical_label": [],
        "mixed_language_labels": [],
        "damage_count_format_issues": [],
    }
    
    label_counts = defaultdict(int)
    label_variants_found = defaultdict(set)
    
    total = 0
    valid_json = 0
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total += 1
            try:
                item = json.loads(line)
            except:
                issues["invalid_json"].append({"line": line_num, "id": "parse_error"})
                continue
            
            item_id = item.get("id", f"line_{line_num}")
            
            # Get GPT response
            gpt_value = ""
            for conv in item.get("conversations", []):
                if conv.get("from") == "gpt":
                    gpt_value = conv.get("value", "")
                    break
            
            # Check empty output
            if not gpt_value or not gpt_value.strip():
                issues["empty_output"].append({"line": line_num, "id": item_id})
                continue
            
            # Check truncated JSON
            if gpt_value.strip().startswith("{") and not gpt_value.strip().endswith("}"):
                issues["truncated_json"].append({"line": line_num, "id": item_id})
            
            # Try to parse JSON
            parsed = parse_json_output(gpt_value)
            if parsed is None:
                issues["invalid_json"].append({"line": line_num, "id": item_id, "snippet": gpt_value[:100]})
                continue
            
            valid_json += 1
            
            # Check disaster_type
            disaster_type = parsed.get("disaster_type", "")
            if disaster_type:
                label_counts[disaster_type] += 1
                
                # Check if canonical
                is_canonical = False
                for canonical, variants in LABEL_VARIANTS.items():
                    if disaster_type.lower() in [v.lower() for v in variants]:
                        label_variants_found[canonical].add(disaster_type)
                        is_canonical = True
                        break
                
                if not is_canonical:
                    issues["non_canonical_label"].append({
                        "line": line_num, 
                        "id": item_id, 
                        "label": disaster_type
                    })
            
            # Check damage_count format
            damage_count = parsed.get("damage_count")
            if damage_count is not None:
                if isinstance(damage_count, str):
                    # String format - check if it's just a number
                    if not damage_count.strip().isdigit():
                        # Contains non-digit characters
                        pass  # This is actually common, not necessarily an issue
                elif isinstance(damage_count, dict):
                    issues["damage_count_format_issues"].append({
                        "line": line_num,
                        "id": item_id,
                        "format": "dict",
                        "value": str(damage_count)[:50]
                    })
    
    # Print report
    print("="*60)
    print("APPROVED_DATASET.JSONL QUALITY AUDIT")
    print("="*60)
    print(f"\nTotal samples: {total}")
    print(f"Valid JSON: {valid_json} ({valid_json/total:.1%})")
    
    print("\n" + "-"*40)
    print("ISSUE SUMMARY")
    print("-"*40)
    
    print(f"\n1. Empty output: {len(issues['empty_output'])} samples")
    if issues['empty_output'][:5]:
        print(f"   Examples: {[e['id'] for e in issues['empty_output'][:5]]}")
    
    print(f"\n2. Truncated JSON: {len(issues['truncated_json'])} samples")
    if issues['truncated_json'][:5]:
        print(f"   Examples: {[e['id'] for e in issues['truncated_json'][:5]]}")
    
    print(f"\n3. Invalid JSON: {len(issues['invalid_json'])} samples")
    if issues['invalid_json'][:5]:
        print(f"   Examples: {[e['id'] for e in issues['invalid_json'][:5]]}")
    
    print(f"\n4. Non-canonical labels: {len(issues['non_canonical_label'])} samples")
    if issues['non_canonical_label'][:10]:
        labels = set(e['label'] for e in issues['non_canonical_label'])
        print(f"   Unique labels found: {labels}")
        for e in issues['non_canonical_label'][:5]:
            print(f"   - {e['id']}: '{e['label']}'")
    
    print(f"\n5. damage_count as dict: {len(issues['damage_count_format_issues'])} samples")
    
    print("\n" + "-"*40)
    print("LABEL DISTRIBUTION")
    print("-"*40)
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {label}: {count}")
    
    print("\n" + "-"*40)
    print("LABEL VARIANTS FOUND")
    print("-"*40)
    for canonical, variants in label_variants_found.items():
        print(f"  {canonical}: {variants}")
    
    # Save detailed report
    report_path = Path(r"C:\Users\34791\Desktop\数据集\data\approved_dataset_audit.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total": total,
            "valid_json": valid_json,
            "issues": issues,
            "label_counts": dict(label_counts)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    main()
