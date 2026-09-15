 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')
 
# 设置中文字体（如需中文标签）与绘图风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
 
print("=" * 70)
print("改进型GM(1,1)灰色预测模型 —— 自主复现")
print("=" * 70)
 
# ==============================================================================
# 第一部分: 数据准备 —— 基于河南7·20特大暴雨的模拟数据
# ==============================================================================
print("\n[Step 1] 数据准备: 生成基于河南7·20模式的模拟数据")
print("-" * 70)
 
# 模拟数据说明:
# 基于2021年河南7·20特大暴雨的灾害 progression pattern:
# Day 1-2: 暴雨初期,转移人口快速增加
# Day 3-5: 暴雨 peak,转移人口指数级增长  
# Day 6-7: 救援展开,增速放缓但总量持续上升
# 单位: 万人 (10,000 persons)
 
dates = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
x0_raw = np.array([12.8, 22.3, 35.6, 52.1, 74.8, 102.4, 134.6])
 
# 添加真实灾害场景的随机波动（模拟信息上报延迟、救援节奏波动等）
np.random.seed(42)
noise = np.random.normal(0, x0_raw * 0.03)  # 3%随机波动
x0 = x0_raw + noise
x0 = np.round(x0, 2)
 
print(f"原始序列 x0 (含3%随机波动): {x0}")
print(f"数据长度 n = {len(x0)}")
print(f"序列均值: {np.mean(x0):.2f} 万人")
print(f"序列标准差: {np.std(x0):.2f} 万人")
 
# 训练集/测试集划分: 前5天训练,后2天验证
train = x0[:5]
test = x0[5:]
print(f"\n训练集 (Day 1-5): {train}")
print(f"测试集 (Day 6-7): {test}")
 
# ==============================================================================
# 第二部分: 标准GM(1,1)模型 —— 从零手写实现
# ==============================================================================
print("\n[Step 2] 标准GM(1,1)模型实现")
print("-" * 70)
 
def standard_gm11(x0, predict_steps=2):
    """
    标准GM(1,1)模型实现
 
    算法步骤 (自主代码实现):
    1. AGO累加生成
    2. 构造背景值矩阵B和向量Y
    3. 最小二乘法估计参数[a, b]
    4. 求解白化方程
    5. IAGO累减还原
    6. 预测
 
    参数:
        x0: 原始非负序列
        predict_steps: 预测步数
    返回:
        x0_fit: 拟合值
        x0_pred: 预测值
        a, b: 模型参数
        mape: 平均绝对百分比误差
    """
    n = len(x0)
 
    # Step 1: AGO (Accumulated Generating Operation)
    x1 = np.cumsum(x0)
    print(f"  [AGO] 累加序列 x1: {np.round(x1, 2)}")
 
    # Step 2: 构造矩阵B和向量Y
    B = np.zeros((n - 1, 2))
    Y = np.zeros(n - 1)
    for i in range(n - 1):
        B[i, 0] = -0.5 * (x1[i] + x1[i + 1])  # 背景值 z^(1)(k+1)
        B[i, 1] = 1
        Y[i] = x0[i + 1]
 
    print(f"  [矩阵B] 形状: {B.shape}")
    print(f"  [向量Y] 形状: {Y.shape}")
 
    # Step 3: 最小二乘估计 [a, b]^T = (B^T·B)^(-1)·B^T·Y
    params = np.linalg.lstsq(B, Y, rcond=None)[0]
    a, b = params[0], params[1]
 
    print(f"  [参数估计] 发展系数 a = {a:.6f}")
    print(f"  [参数估计] 灰色作用量 b = {b:.6f}")
 
    # Step 4 & 5: 白化方程求解 + IAGO还原
    x1_fit = np.zeros(n)
    x0_fit = np.zeros(n)
 
    for k in range(n):
        x1_fit[k] = (x0[0] - b / a) * np.exp(-a * k) + b / a
 
    x0_fit[0] = x0[0]
    for k in range(1, n):
        x0_fit[k] = x1_fit[k] - x1_fit[k - 1]
 
    # Step 6: 预测
    x0_pred = np.zeros(predict_steps)
    for i in range(predict_steps):
        k = n + i
        x1_pred_k = (x0[0] - b / a) * np.exp(-a * k) + b / a
        x1_pred_k_prev = (x0[0] - b / a) * np.exp(-a * (k - 1)) + b / a
        x0_pred[i] = x1_pred_k - x1_pred_k_prev
 
    # 精度评估
    residuals = x0 - x0_fit
    mape = np.mean(np.abs(residuals / x0)) * 100
    rmse = np.sqrt(np.mean(residuals ** 2))
 
    return x0_fit, x0_pred, a, b, mape, rmse
 
# 运行标准GM(1,1)
fit_std, pred_std, a_std, b_std, mape_std, rmse_std = standard_gm11(train, predict_steps=2)
 
print(f"\n  [标准GM(1,1)结果]")
print(f"  拟合值: {np.round(fit_std, 2)}")
print(f"  预测值 (Day 6-7): {np.round(pred_std, 2)}")
print(f"  训练集MAPE: {mape_std:.2f}%")
print(f"  训练集RMSE: {rmse_std:.2f}")
 
# ==============================================================================
# 第三部分: 一阶缓冲算子 —— 自主设计与实现
# ==============================================================================
print("\n[Step 3] 一阶缓冲算子 (First-Order Buffer Operator)")
print("-" * 70)
 
def buffer_operator(x, alpha):
    """
    一阶缓冲算子 (Weakening Buffer Operator)
 
    设计思路 (自主):
    灾害数据存在显著随机波动(降雨强度变化、救援进度差异、信息上报延迟等),
    直接建模会引入噪声。缓冲算子通过加权平滑,削弱异常波动,保留长期趋势。
 
    公式:
        x_buffer(1) = x(1)
        x_buffer(k) = alpha * x(k) + (1-alpha) * x_buffer(k-1), k >= 2
 
    参数:
        x: 原始序列
        alpha: 缓冲系数, 0 < alpha < 1
               alpha越大,平滑效果越强
    """
    n = len(x)
    x_buf = np.zeros(n)
    x_buf[0] = x[0]
    for i in range(1, n):
        x_buf[i] = alpha * x[i] + (1 - alpha) * x_buf[i - 1]
    return x_buf
 
# 演示缓冲算子效果
demo_alpha = 0.85
x_buffered_demo = buffer_operator(train, demo_alpha)
 
print(f"  原始序列:     {np.round(train, 2)}")
print(f"  缓冲后序列 (α={demo_alpha}): {np.round(x_buffered_demo, 2)}")
print(f"  标准差变化: {np.std(train):.2f} -> {np.std(x_buffered_demo):.2f}")
 
# ==============================================================================
# 第四部分: 改进GM(1,1)模型 —— 缓冲算子 + GM(1,1)
# ==============================================================================
print("\n[Step 4] 改进GM(1,1)模型实现")
print("-" * 70)
 
def improved_gm11(x0, alpha, predict_steps=2):
    """
    改进型GM(1,1)模型
 
    改进策略 (自主设计):
    1. 先对原始数据应用一阶缓冲算子,降低随机波动
    2. 对缓冲后的数据建立标准GM(1,1)模型
    3. 通过比例系数将拟合/预测值映射回原始尺度
 
    参数:
        x0: 原始序列
        alpha: 缓冲系数
        predict_steps: 预测步数
    """
    # Step 1: 缓冲算子平滑
    x0_buffered = buffer_operator(x0, alpha)
 
    # Step 2: 对缓冲数据建立GM(1,1)
    fit_buf, pred_buf, a, b, _, _ = standard_gm11(x0_buffered, predict_steps)
 
    # Step 3: 映射回原始尺度 (比例调整法)
    ratio = np.mean(x0 / x0_buffered)
    x0_fit = fit_buf * ratio
    x0_pred = pred_buf * ratio
 
    # 重新计算原始尺度下的精度
    residuals = x0 - x0_fit
    mape = np.mean(np.abs(residuals / x0)) * 100
    rmse = np.sqrt(np.mean(residuals ** 2))
 
    return x0_fit, x0_pred, x0_buffered, a, b, mape, rmse
 
# ==============================================================================
# 第五部分: 缓冲系数优化 —— 网格搜索 (自主设计)
# ==============================================================================
print("\n[Step 5] 缓冲系数 α 优化 (Grid Search)")
print("-" * 70)
 
def optimize_alpha(x0, predict_steps=2, alpha_range=None):
    """
    网格搜索最优缓冲系数
 
    搜索策略 (自主):
    在 (0.1, 0.98) 区间内以步长0.01进行网格搜索,
    目标函数为训练集MAPE最小化。
    """
    if alpha_range is None:
        alpha_range = np.arange(0.1, 0.98, 0.01)
 
    best_alpha = 0.5
    best_mape = float('inf')
    history = []
 
    for alpha in alpha_range:
        _, _, _, _, _, mape, _ = improved_gm11(x0, alpha, predict_steps)
        history.append((alpha, mape))
        if mape < best_mape:
            best_mape = mape
            best_alpha = alpha
 
    return best_alpha, best_mape, history
 
best_alpha, best_mape, search_history = optimize_alpha(train, predict_steps=2)
 
print(f"  搜索范围: α ∈ [0.10, 0.97], 步长=0.01")
print(f"  最优缓冲系数: α = {best_alpha:.2f}")
print(f"  最优训练集MAPE: {best_mape:.2f}%")
print(f"  对应预测精度: {100 - best_mape:.2f}%")
 
# ==============================================================================
# 第六部分: 运行改进模型并对比
# ==============================================================================
print("\n[Step 6] 运行改进模型并与标准模型对比")
print("-" * 70)
 
fit_imp, pred_imp, buffered, a_imp, b_imp, mape_imp, rmse_imp = improved_gm11(
    train, best_alpha, predict_steps=2
)
 
print(f"\n  [改进GM(1,1)结果]")
print(f"  缓冲后序列: {np.round(buffered, 2)}")
print(f"  拟合值: {np.round(fit_imp, 2)}")
print(f"  预测值 (Day 6-7): {np.round(pred_imp, 2)}")
print(f"  训练集MAPE: {mape_imp:.2f}%")
print(f"  训练集RMSE: {rmse_imp:.2f}")
 
# 测试集精度
test_mape_std = np.mean(np.abs(test - pred_std) / test) * 100
test_mape_imp = np.mean(np.abs(test - pred_imp) / test) * 100
 
print(f"\n  [测试集对比 (Day 6-7)]")
print(f"  {'日期':<10} {'实际值':<10} {'标准GM预测':<12} {'改进GM预测':<12}")
for i in range(2):
    print(f"  {dates[5+i]:<10} {test[i]:<10.2f} {pred_std[i]:<12.2f} {pred_imp[i]:<12.2f}")
 
print(f"\n  [精度对比总结]")
print(f"  指标                标准GM(1,1)    改进GM(1,1)")
print(f"  {'-'*50}")
print(f"  训练集MAPE          {mape_std:>6.2f}%        {mape_imp:>6.2f}%")
print(f"  训练集精度          {100-mape_std:>6.2f}%        {100-mape_imp:>6.2f}%")
print(f"  测试集MAPE          {test_mape_std:>6.2f}%        {test_mape_imp:>6.2f}%")
print(f"  测试集精度          {100-test_mape_std:>6.2f}%        {100-test_mape_imp:>6.2f}%")
 
# ==============================================================================
# 第七部分: 可视化 —— 自主绘制
# ==============================================================================
print("\n[Step 7] 生成可视化图表")
print("-" * 70)
 
RED = '#C00000'
BLUE = '#4472C4'
GRAY = '#808080'
LIGHT_RED = '#F2DCDC'
LIGHT_BLUE = '#D6E3F8'
LIGHT_GRAY = '#E0E0E0'
 
fig = plt.figure(figsize=(14, 10))
 
# 子图1: 缓冲算子效果
ax1 = fig.add_subplot(2, 2, 1)
days_train = np.arange(1, 6)
ax1.plot(days_train, train, 'o-', color=GRAY, linewidth=2.5, markersize=10, 
         label='Original Data', zorder=5)
ax1.plot(days_train, buffered, 's-', color=BLUE, linewidth=2.5, markersize=8, 
         label=f'Buffered (α={best_alpha:.2f})', zorder=4)
for i in [2, 4]:
    ax1.annotate('', xy=(days_train[i-1], buffered[i-1]), 
                xytext=(days_train[i-1], train[i-1]),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2))
ax1.text(3, np.mean(train)*1.1, 'Buffer\nSmooths\nFluctuations', 
        fontsize=9, color=RED, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor=LIGHT_RED, edgecolor=RED, alpha=0.8))
ax1.set_xlabel('Time (Days)', fontweight='bold')
ax1.set_ylabel('Displaced Population (10k)', fontweight='bold')
ax1.set_title('(a) Buffer Operator Effect', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
 
# 子图2: 网格搜索过程
ax2 = fig.add_subplot(2, 2, 2)
alphas, mapes = zip(*search_history)
ax2.plot(alphas, mapes, '-', color=BLUE, linewidth=1.5)
ax2.axvline(x=best_alpha, color=RED, linestyle='--', linewidth=2, 
           label=f'Optimal α={best_alpha:.2f}')
ax2.scatter([best_alpha], [best_mape], color=RED, s=100, zorder=5)
ax2.set_xlabel('Buffer Coefficient α', fontweight='bold')
ax2.set_ylabel('Training MAPE (%)', fontweight='bold')
ax2.set_title('(b) Grid Search for Optimal α', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
 
# 子图3: 拟合对比
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(days_train, train, 'o-', color=GRAY, linewidth=2.5, markersize=10, 
         label='Actual (Training)', zorder=5)
ax3.plot(days_train, fit_std, 'v:', color=BLUE, linewidth=1.5, markersize=7, 
         label=f'Standard GM(1,1) (MAPE={mape_std:.1f}%)', alpha=0.7)
ax3.plot(days_train, fit_imp, '^-', color=RED, linewidth=2, markersize=8, 
         label=f'Improved GM(1,1) (MAPE={mape_imp:.1f}%)', zorder=4)
ax3.set_xlabel('Time (Days)', fontweight='bold')
ax3.set_ylabel('Displaced Population (10k)', fontweight='bold')
ax3.set_title('(c) Model Fitting Comparison', fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
 
# 子图4: 预测对比
ax4 = fig.add_subplot(2, 2, 4)
days_test = np.arange(6, 8)
width = 0.25
x_pos = np.arange(2)
bars1 = ax4.bar(x_pos - width, test, width, color=GRAY, edgecolor='white', label='Actual')
bars2 = ax4.bar(x_pos, pred_std, width, color=BLUE, edgecolor='white', 
               label=f'Standard GM (MAPE={test_mape_std:.1f}%)')
bars3 = ax4.bar(x_pos + width, pred_imp, width, color=RED, edgecolor='white', 
               label=f'Improved GM (MAPE={test_mape_imp:.1f}%)')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(['Day 6', 'Day 7'])
ax4.set_ylabel('Displaced Population (10k)', fontweight='bold')
ax4.set_title('(d) Prediction on Test Set', fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
 
fig.suptitle('Figure: Independent Reproduction of Improved GM(1,1) Model\n'
             'Author: Jian Yang | Date: 2026-08-26', 
             fontsize=14, fontweight='bold', y=0.98)
 
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('gm11_reproduction_result.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
 
print("\n图表已保存: gm11_reproduction_result.png")
 
# ==============================================================================
# 第八部分: 结果保存
# ==============================================================================
print("\n[Step 8] 保存模型参数与结果")
print("-" * 70)
 
results = {
    'model': 'Improved GM(1,1)',
    'author': 'Jian Yang',
    'date': '2026-08-26',
    'standard_gm11': {
        'a': float(a_std),
        'b': float(b_std),
        'mape_train': float(mape_std),
        'rmse_train': float(rmse_std),
        'mape_test': float(test_mape_std)
    },
    'improved_gm11': {
        'alpha': float(best_alpha),
        'a': float(a_imp),
        'b': float(b_imp),
        'mape_train': float(mape_imp),
        'rmse_train': float(rmse_imp),
        'mape_test': float(test_mape_imp),
        'accuracy_train': float(100 - mape_imp)
    },
    'data': {
        'dates': dates,
        'raw_sequence': x0.tolist(),
        'train': train.tolist(),
        'test': test.tolist()
    }
}
 
import json
with open('gm11_model_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
 
print("结果已保存: gm11_model_results.json")
print("\n" + "=" * 70)
print("复现完成! 核心结论:")
print(f"  - 最优缓冲系数 α = {best_alpha:.2f}")
print(f"  - 改进模型训练精度 = {100-mape_imp:.2f}%")
print(f"  - 改进模型测试精度 = {100-test_mape_imp:.2f}%")
print("=" * 70)
 