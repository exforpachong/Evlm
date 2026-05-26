"""
配对显著性检验脚本

验证 Zero-shot vs Fine-tuned 的准确率差异是否具有统计显著性。

输出:
- McNemar test (精确二项检验)
- Paired bootstrap 95% CI
- 详细配对分析表
"""

import json
import numpy as np
from scipy import stats
from collections import defaultdict
import os

def load_results(filepath):
    """加载评测结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_prediction_dict(results):
    """创建 {id: prediction} 字典"""
    pred_dict = {}
    for r in results:
        sample_id = r['id']
        ground_truth = r['ground_truth']
        prediction = r.get('prediction')
        valid_json = r.get('valid_json', True)
        
        # Strict accuracy: invalid JSON 或无 prediction 视为错误
        if not valid_json or prediction is None:
            pred_dict[sample_id] = {'correct': False, 'prediction': None, 'ground_truth': ground_truth}
        else:
            correct = (prediction == ground_truth)
            pred_dict[sample_id] = {'correct': correct, 'prediction': prediction, 'ground_truth': ground_truth}
    
    return pred_dict

def mcnemar_test(b, c):
    """
    McNemar test (精确二项检验版本)
    
    b: Zero-shot 正确但 Fine-tuned 错误的样本数
    c: Fine-tuned 正确但 Zero-shot 错误的样本数
    
    H0: P(b) = P(c) (两模型性能无差异)
    H1: P(b) ≠ P(c) (两模型性能有差异)
    """
    if b + c == 0:
        return None, None
    
    # 精确二项检验 (推荐用于小样本)
    # 在 H0 下，b ~ Binomial(b+c, 0.5)
    # 双侧 p-value
    n = b + c
    p_value = 2 * min(
        stats.binom.cdf(min(b, c), n, 0.5),
        1 - stats.binom.cdf(min(b, c) - 1, n, 0.5)
    )
    
    # 近似卡方检验 (大样本)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0
    chi2_p = 1 - stats.chi2.cdf(chi2, df=1)
    
    return p_value, chi2_p

def paired_bootstrap_ci(zeroshot_correct, finetuned_correct, n_bootstrap=10000, confidence=0.95):
    """
    Paired bootstrap confidence interval for accuracy difference
    
    Returns: (lower, upper, mean_diff)
    """
    n = len(zeroshot_correct)
    diffs = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        zs_acc = np.mean(zeroshot_correct[indices])
        ft_acc = np.mean(finetuned_correct[indices])
        diffs.append(ft_acc - zs_acc)
    
    diffs = np.array(diffs)
    alpha = 1 - confidence
    lower = np.percentile(diffs, alpha / 2 * 100)
    upper = np.percentile(diffs, (1 - alpha / 2) * 100)
    mean_diff = np.mean(diffs)
    
    return lower, upper, mean_diff

def analyze_paired_results(zeroshot_path, finetuned_path):
    """主分析函数"""
    print("=" * 60)
    print("配对显著性检验报告")
    print("=" * 60)
    
    # 加载结果
    zs_results = load_results(zeroshot_path)
    ft_results = load_results(finetuned_path)
    
    print(f"\n[数据]")
    print(f"  Zero-shot 样本数: {len(zs_results)}")
    print(f"  Fine-tuned 样本数: {len(ft_results)}")
    
    # 创建预测字典
    zs_dict = create_prediction_dict(zs_results)
    ft_dict = create_prediction_dict(ft_results)
    
    # 确保样本 ID 匹配
    common_ids = set(zs_dict.keys()) & set(ft_dict.keys())
    print(f"  配对样本数: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("错误: 无配对样本")
        return
    
    # 配对分析
    both_correct = 0
    both_wrong = 0
    zs_only = 0  # Zero-shot 正确, Fine-tuned 错误
    ft_only = 0  # Fine-tuned 正确, Zero-shot 错误
    
    zeroshot_correct_list = []
    finetuned_correct_list = []
    
    error_analysis = {
        'zs_only_correct': [],
        'ft_only_correct': [],
        'both_wrong': []
    }
    
    for sample_id in sorted(common_ids):
        zs_corr = zs_dict[sample_id]['correct']
        ft_corr = ft_dict[sample_id]['correct']
        
        zeroshot_correct_list.append(1 if zs_corr else 0)
        finetuned_correct_list.append(1 if ft_corr else 0)
        
        if zs_corr and ft_corr:
            both_correct += 1
        elif zs_corr and not ft_corr:
            zs_only += 1
            error_analysis['zs_only_correct'].append({
                'id': sample_id,
                'ground_truth': zs_dict[sample_id]['ground_truth'],
                'zs_pred': zs_dict[sample_id]['prediction'],
                'ft_pred': ft_dict[sample_id]['prediction']
            })
        elif not zs_corr and ft_corr:
            ft_only += 1
            error_analysis['ft_only_correct'].append({
                'id': sample_id,
                'ground_truth': zs_dict[sample_id]['ground_truth'],
                'zs_pred': zs_dict[sample_id]['prediction'],
                'ft_pred': ft_dict[sample_id]['prediction']
            })
        else:
            both_wrong += 1
            error_analysis['both_wrong'].append({
                'id': sample_id,
                'ground_truth': zs_dict[sample_id]['ground_truth'],
                'zs_pred': zs_dict[sample_id]['prediction'],
                'ft_pred': ft_dict[sample_id]['prediction']
            })
    
    total = len(common_ids)
    zs_correct = both_correct + zs_only
    ft_correct = both_correct + ft_only
    
    print(f"\n[配对分析表]")
    print("  Zero-shot Correct / Fine-tuned Correct: "
          f"{both_correct}")
    print("  Zero-shot Correct / Fine-tuned Wrong:   "
          f"{zs_only}")
    print("  Zero-shot Wrong   / Fine-tuned Correct: "
          f"{ft_only}")
    print("  Zero-shot Wrong   / Fine-tuned Wrong:   "
          f"{both_wrong}")
    
    print(f"\n[准确率 (Strict 口径)]")
    print(f"  Zero-shot:  {zs_correct}/{total} = {zs_correct/total*100:.2f}%")
    print(f"  Fine-tuned: {ft_correct}/{total} = {ft_correct/total*100:.2f}%")
    print(f"  差异: {(ft_correct - zs_correct)}/{total} = {(ft_correct - zs_correct)/total*100:.2f}%")
    print(f"  净增样本: {ft_only - zs_only} (Fine-tuned 独对 {ft_only} - Zero-shot 独对 {zs_only})")
    
    # McNemar test
    print(f"\n[McNemar 检验]")
    print(f"  b (ZS正确/FT错误): {zs_only}")
    print(f"  c (FT正确/ZS错误): {ft_only}")
    
    exact_p, chi2_p = mcnemar_test(zs_only, ft_only)
    
    if exact_p is not None:
        print(f"\n  精确二项检验 (推荐):")
        print(f"    p-value = {exact_p:.4f}")
        if exact_p < 0.05:
            print(f"    结论: p < 0.05, 差异具有统计显著性")
        else:
            print(f"    结论: p >= 0.05, 差异不具有统计显著性")
        
        if chi2_p is not None:
            print(f"\n  近似卡方检验 (大样本):")
            print(f"    p-value = {chi2_p:.4f}")
    else:
        print(f"  b + c = 0, 无法进行 McNemar 检验")
    
    # Paired bootstrap CI
    zeroshot_correct_arr = np.array(zeroshot_correct_list)
    finetuned_correct_arr = np.array(finetuned_correct_list)
    
    lower, upper, mean_diff = paired_bootstrap_ci(
        zeroshot_correct_arr, finetuned_correct_arr
    )
    
    print(f"\n[Paired Bootstrap 95% CI]")
    print(f"  Accuracy 差异: {mean_diff*100:.2f}%")
    print(f"  95% CI: [{lower*100:.2f}%, {upper*100:.2f}%]")
    
    if lower > 0:
        print(f"  结论: CI 不包含 0, Fine-tuned 显著优于 Zero-shot")
    elif upper < 0:
        print(f"  结论: CI 不包含 0, Zero-shot 显著优于 Fine-tuned")
    else:
        print(f"  结论: CI 包含 0, 无法证明显著差异")
    
    # 论文表述建议
    print(f"\n[论文表述建议]")
    if exact_p is not None and exact_p < 0.05 and lower > 0:
        print(f"  可写: 'LoRA 微调显著提升了灾害识别准确率 (p={exact_p:.3f})'")
        print(f"  可写: '准确率提升 {(ft_correct - zs_correct)/total*100:.2f}% (95% CI: [{lower*100:.2f}%, {upper*100:.2f}%])'")
    else:
        print(f"  建议: 'LoRA 微调带来小幅正向趋势 (+{(ft_correct - zs_correct)/total*100:.2f}%)'")
        p_str = f"{exact_p:.3f}" if exact_p is not None else "N/A"
        print(f"  建议: '配对检验 p={p_str}, 差异未达统计显著性'")
        print(f"  不应写: '显著提升'")
    
    # 错误分析
    print(f"\n[错误样例分析]")
    print(f"\n  Zero-shot 独对样本 ({len(error_analysis['zs_only_correct'])} 个):")
    for e in error_analysis['zs_only_correct'][:5]:
        print(f"    {e['id']}: GT={e['ground_truth']}, ZS={e['zs_pred']}, FT={e['ft_pred']}")
    
    print(f"\n  Fine-tuned 独对样本 ({len(error_analysis['ft_only_correct'])} 个):")
    for e in error_analysis['ft_only_correct'][:5]:
        print(f"    {e['id']}: GT={e['ground_truth']}, ZS={e['zs_pred']}, FT={e['ft_pred']}")
    
    # 保存结果
    output = {
        'total_samples': int(total),
        'both_correct': int(both_correct),
        'both_wrong': int(both_wrong),
        'zs_only_correct': int(zs_only),
        'ft_only_correct': int(ft_only),
        'zs_accuracy': float(zs_correct / total),
        'ft_accuracy': float(ft_correct / total),
        'accuracy_diff': float((ft_correct - zs_correct) / total),
        'net_gain_samples': int(ft_only - zs_only),
        'mcnemar_exact_p': float(exact_p) if exact_p is not None else None,
        'mcnemar_chi2_p': float(chi2_p) if chi2_p is not None else None,
        'bootstrap_ci_lower': float(lower),
        'bootstrap_ci_upper': float(upper),
        'bootstrap_mean_diff': float(mean_diff),
        'significant_at_0.05': bool(exact_p is not None and exact_p < 0.05 and lower > 0),
        'error_analysis': error_analysis
    }
    
    output_path = 'eval_results_v3/paired_significance_test.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[结果已保存]")
    print(f"  {output_path}")
    
    return output

if __name__ == '__main__':
    zeroshot_path = 'eval_results_v3/zeroshot_results.json'
    finetuned_path = 'eval_results_v3/finetuned_results.json'
    
    analyze_paired_results(zeroshot_path, finetuned_path)
