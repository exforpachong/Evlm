#!/usr/bin/env python3
"""
Build canonical clean dataset - only samples with valid parseable JSON.
This is the benchmark-ready dataset for structured output evaluation.

Outputs:
- dataset_*_canonical_clean.json: Only samples with valid JSON and canonical labels
- canonical_clean_statistics.json: Statistics of the clean dataset
"""

import json
import re
from pathlib import Path
from collections import defaultdict

CANONICAL_LABELS = ["flood", "earthquake", "fire", "landslide", "windstorm_or_typhoon", "other"]

def parse_json_from_text(text: str) -> dict | None:
    """Extract and parse JSON from assistant message."""
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Find JSON object
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return None
    
    json_str = text[start_idx:end_idx+1]
    
    try:
        parsed = json.loads(json_str)
        return parsed
    except json.JSONDecodeError:
        return None

def validate_sample(item: dict) -> tuple[bool, dict | None, str]:
    """
    Validate a single sample.
    Returns: (is_valid, parsed_json, error_reason)
    """
    conversations = item.get('conversations', [])
    
    # Find assistant message
    assistant_msg = None
    for conv in conversations:
        if conv.get('from') == 'assistant':
            assistant_msg = conv.get('value', '')
            break
    
    if not assistant_msg:
        return False, None, "empty_assistant_message"
    
    # Parse JSON
    parsed = parse_json_from_text(assistant_msg)
    
    if not parsed:
        return False, None, "json_parse_error"
    
    # Check disaster_type
    disaster_type = parsed.get('disaster_type')
    if not disaster_type:
        return False, parsed, "missing_disaster_type"
    
    # Verify canonical label
    if disaster_type not in CANONICAL_LABELS:
        return False, parsed, f"non_canonical_label:{disaster_type}"
    
    return True, parsed, "ok"

def process_split(input_path: str, output_path: str, split_name: str) -> dict:
    """Process a single split and generate clean version."""
    print(f"\nProcessing {split_name}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clean_data = []
    stats = defaultdict(lambda: defaultdict(int))
    error_samples = []
    
    for item in data:
        stats['total']['count'] += 1
        
        is_valid, parsed, reason = validate_sample(item)
        
        if is_valid:
            clean_data.append(item)
            stats['valid']['count'] += 1
            
            # Track label distribution
            if parsed and 'disaster_type' in parsed:
                stats['labels'][parsed['disaster_type']] += 1
        else:
            stats['errors'][reason] += 1
            error_samples.append({
                'id': item.get('id', 'unknown'),
                'reason': reason
            })
    
    # Save clean data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    
    print(f"  Total: {stats['total']['count']}")
    print(f"  Valid: {stats['valid']['count']}")
    print(f"  Errors: {dict(stats['errors'])}")
    print(f"  Label distribution: {dict(stats['labels'])}")
    
    return {
        'total': stats['total']['count'],
        'valid': stats['valid']['count'],
        'labels': dict(stats['labels']),
        'errors': dict(stats['errors']),
        'error_samples': error_samples[:20]  # Keep first 20 for debugging
    }

def main():
    base_path = Path(__file__).parent.parent / 'data'
    
    splits = ['train', 'val', 'test']
    all_stats = {}
    
    print("="*60)
    print("Building Canonical Clean Dataset")
    print("="*60)
    
    for split in splits:
        input_file = base_path / f'dataset_{split}_canonical.json'
        output_file = base_path / f'dataset_{split}_canonical_clean.json'
        
        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping...")
            continue
        
        all_stats[split] = process_split(str(input_file), str(output_file), split)
    
    # Save statistics
    stats_path = base_path / 'canonical_clean_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\n{'Split':<10} {'Total':>8} {'Valid':>8} {'Rate':>10}")
    print("-" * 40)
    
    for split in splits:
        if split in all_stats:
            s = all_stats[split]
            rate = s['valid'] / s['total'] * 100 if s['total'] > 0 else 0
            print(f"{split:<10} {s['total']:>8} {s['valid']:>8} {rate:>9.1f}%")
    
    # Print label distribution for test set
    if 'test' in all_stats:
        print("\nTest Set Label Distribution (Clean):")
        test_labels = all_stats['test'].get('labels', {})
        for label in CANONICAL_LABELS:
            count = test_labels.get(label, 0)
            print(f"  {label}: {count}")
    
    print(f"\nStatistics saved to: {stats_path}")

if __name__ == '__main__':
    main()
