#!/usr/bin/env python3
"""
Build classification full dataset with filename-prefix fallback.
This ensures 100% coverage of the test set for classification benchmark.

For samples with valid JSON: use canonical label
For samples without valid JSON: use filename prefix as fallback

Outputs:
- dataset_*_classification_full.json: All samples with disaster_type
- classification_full_statistics.json: Statistics including label sources
"""

import json
import re
from pathlib import Path
from collections import defaultdict

CANONICAL_LABELS = ["flood", "earthquake", "fire", "landslide", "windstorm_or_typhoon", "other"]

# File prefix to canonical label mapping
FILENAME_PREFIX_MAP = {
    "flood": "flood",
    "flood_洪涝": "flood",
    "earthquake": "earthquake",
    "earthquake_地震": "earthquake",
    "fire": "fire",
    "fire_火灾": "fire",
    "landslide": "landslide",
    "landslide_滑坡": "landslide",
    "typhoon": "windstorm_or_typhoon",
    "typhoon_台风": "windstorm_or_typhoon",
    "windstorm": "windstorm_or_typhoon",
    "other": "other",
    "unknown": "other",
}

def parse_json_from_text(text: str) -> dict | None:
    """Extract and parse JSON from assistant message."""
    if not text or not text.strip():
        return None
    
    text = text.strip()
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return None
    
    json_str = text[start_idx:end_idx+1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def infer_label_from_filename(item_id: str) -> tuple[str, str]:
    """
    Infer disaster_type from filename/id prefix.
    Returns: (canonical_label, matched_prefix)
    """
    # Extract prefix from id (e.g., "flood_洪涝_59" -> "flood_洪涝")
    parts = item_id.split('_')
    
    # Try multi-part prefixes first
    for i in range(min(3, len(parts))):
        prefix = '_'.join(parts[:i+1])
        if prefix in FILENAME_PREFIX_MAP:
            return FILENAME_PREFIX_MAP[prefix], prefix
    
    # Try single part
    for part in parts:
        part_lower = part.lower()
        if part_lower in FILENAME_PREFIX_MAP:
            return FILENAME_PREFIX_MAP[part_lower], part
    
    # Fuzzy matching
    id_lower = item_id.lower()
    if any(kw in id_lower for kw in ["flood", "洪涝", "洪水"]):
        return "flood", "fuzzy:flood"
    if any(kw in id_lower for kw in ["earthquake", "地震"]):
        return "earthquake", "fuzzy:earthquake"
    if any(kw in id_lower for kw in ["fire", "火灾", "山火"]):
        return "fire", "fuzzy:fire"
    if any(kw in id_lower for kw in ["landslide", "滑坡", "泥石流"]):
        return "landslide", "fuzzy:landslide"
    if any(kw in id_lower for kw in ["typhoon", "台风", "windstorm", "风"]):
        return "windstorm_or_typhoon", "fuzzy:typhoon"
    
    return "other", "fallback"

def process_sample(item: dict) -> dict:
    """
    Process a single sample and add disaster_type with label_source.
    Returns: modified item with classification metadata
    """
    item_id = item.get('id', '')
    conversations = item.get('conversations', [])
    
    # Try to parse from assistant message
    assistant_msg = None
    for conv in conversations:
        if conv.get('from') == 'assistant':
            assistant_msg = conv.get('value', '')
            break
    
    parsed = parse_json_from_text(assistant_msg) if assistant_msg else None
    
    result = item.copy()
    classification_info = {}
    
    if parsed and 'disaster_type' in parsed:
        # Use canonical label from JSON
        label = parsed['disaster_type']
        if label in CANONICAL_LABELS:
            classification_info = {
                'disaster_type': label,
                'label_source': 'canonical',
                'confidence': 'high'
            }
        else:
            # Non-canonical label, try to normalize
            label_lower = label.lower()
            if label_lower in CANONICAL_LABELS:
                classification_info = {
                    'disaster_type': label_lower,
                    'label_source': 'canonical_normalized',
                    'original_label': label,
                    'confidence': 'medium'
                }
            else:
                # Fall back to filename
                inferred, prefix = infer_label_from_filename(item_id)
                classification_info = {
                    'disaster_type': inferred,
                    'label_source': 'filename_fallback',
                    'original_label': label,
                    'matched_prefix': prefix,
                    'confidence': 'low'
                }
    else:
        # No valid JSON, use filename prefix
        inferred, prefix = infer_label_from_filename(item_id)
        classification_info = {
            'disaster_type': inferred,
            'label_source': 'filename_fallback',
            'matched_prefix': prefix,
            'confidence': 'low'
        }
    
    result['classification'] = classification_info
    return result

def process_split(input_path: str, output_path: str, split_name: str) -> dict:
    """Process a single split and generate classification full version."""
    print(f"\nProcessing {split_name}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    stats = defaultdict(lambda: defaultdict(int))
    label_sources = defaultdict(int)
    low_confidence_samples = []
    
    for item in data:
        processed = process_sample(item)
        processed_data.append(processed)
        
        stats['total']['count'] += 1
        
        classification = processed.get('classification', {})
        label = classification.get('disaster_type', 'other')
        source = classification.get('label_source', 'unknown')
        confidence = classification.get('confidence', 'unknown')
        
        stats['labels'][label] += 1
        label_sources[source] += 1
        
        if confidence == 'low':
            low_confidence_samples.append({
                'id': item.get('id', 'unknown'),
                'disaster_type': label,
                'label_source': source
            })
    
    # Save processed data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"  Total: {stats['total']['count']}")
    print(f"  Label distribution: {dict(stats['labels'])}")
    print(f"  Label sources: {dict(label_sources)}")
    print(f"  Low confidence samples: {len(low_confidence_samples)}")
    
    return {
        'total': stats['total']['count'],
        'labels': dict(stats['labels']),
        'label_sources': dict(label_sources),
        'low_confidence_count': len(low_confidence_samples),
        'low_confidence_samples': low_confidence_samples[:50]  # Keep first 50 for review
    }

def main():
    base_path = Path(__file__).parent.parent / 'data'
    
    splits = ['train', 'val', 'test']
    all_stats = {}
    
    print("="*60)
    print("Building Classification Full Dataset")
    print("="*60)
    
    for split in splits:
        input_file = base_path / f'dataset_{split}_canonical.json'
        output_file = base_path / f'dataset_{split}_classification_full.json'
        
        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping...")
            continue
        
        all_stats[split] = process_split(str(input_file), str(output_file), split)
    
    # Save statistics
    stats_path = base_path / 'classification_full_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\n{'Split':<10} {'Total':>8} {'Canonical':>12} {'Fallback':>12}")
    print("-" * 50)
    
    for split in splits:
        if split in all_stats:
            s = all_stats[split]
            sources = s.get('label_sources', {})
            canonical = sources.get('canonical', 0) + sources.get('canonical_normalized', 0)
            fallback = sources.get('filename_fallback', 0)
            print(f"{split:<10} {s['total']:>8} {canonical:>12} {fallback:>12}")
    
    # Print test set details
    if 'test' in all_stats:
        print("\nTest Set Label Distribution (Full):")
        test_labels = all_stats['test'].get('labels', {})
        for label in CANONICAL_LABELS:
            count = test_labels.get(label, 0)
            print(f"  {label}: {count}")
        
        print("\nTest Set Label Sources:")
        test_sources = all_stats['test'].get('label_sources', {})
        for source, count in test_sources.items():
            print(f"  {source}: {count}")
        
        print(f"\nLow confidence samples (need manual review): {all_stats['test'].get('low_confidence_count', 0)}")
    
    print(f"\nStatistics saved to: {stats_path}")

if __name__ == '__main__':
    main()
