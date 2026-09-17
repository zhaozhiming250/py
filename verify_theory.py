"""
思考题 1 & 2 的数值验证
------------------------------------------------------------
Q1: 一元 OLS 中 sum(e_i) = 0 且 sum(x_i * e_i) = 0 (正规方程的直接推论)
Q2: 加入纯随机无关特征后, R^2 单调不减, 而调整 R^2 大概率下降;
    并验证"调整 R^2 上升 <=> 新增变量 |t| > 1"这一精确判据。
"""

import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(20240517)

# =================================================== Q1: 正规方程数值验证
print("=" * 74)
print("Q1  一元线性回归: sum(e_i)=0 与 sum(x_i e_i)=0 的数值验证")
print("=" * 74)
n = 50
x = rng.normal(10, 3, n)
y = 4.0 + 1.5 * x + rng.normal(0, 2, n)

res = sm.OLS(y, sm.add_constant(x)).fit()
e = res.resid
print(f"截距估计 b0        = {res.params[0]:.6f}")
print(f"斜率估计 b1        = {res.params[1]:.6f}")
print(f"sum(e_i)           = {e.sum():.3e}     (理论上恰为 0)")
print(f"sum(x_i * e_i)     = {(x * e).sum():.3e}     (理论上恰为 0)")
print(f"sum(yhat_i) == sum(y_i): {np.isclose(res.fittedvalues.sum(), y.sum())}")
print(f"corr(x, e)         = {np.corrcoef(x, e)[0, 1]:.3e}   (理论上为 0)")
print(f"样本协方差 cov(x,e) = {np.cov(x, e, ddof=1)[0, 1]:.3e}")
print(f"x 与 yhat 的相关系数 = {np.corrcoef(x, res.fittedvalues)[0, 1]:.6f} (应等于 corr(x,y) 的绝对值)")

# 反例: 无截距模型下 sum(e_i) 不再为 0
res0 = sm.OLS(y, x).fit()   # 强制过原点, 无截距
print(f"\n[反例] 去掉截距后 sum(e_i) = {res0.resid.sum():.4f}  (不再为 0!)")
print(f"[反例] 去掉截距后 sum(x_i e_i) = {(x * res0.resid).sum():.3e}  (仍为 0)")
print("=> 结论: sum(e_i)=0 依赖于模型含截距项; sum(x_i e_i)=0 直接来自斜率的一阶条件。")

# ============================================ Q2: 加入随机噪声特征的影响
print("\n" + "=" * 74)
print("Q2  加入纯随机无关特征: R^2 与调整 R^2 的行为")
print("=" * 74)


def fit(Xd, yv):
    m = sm.OLS(yv, sm.add_constant(Xd)).fit()
    return m


n, reps = 200, 20000
n_feat = 3
R2_up = R2_same = R2_down = 0
adj_up = 0
dR2, dadj = [], []
t_consistency = True

rng2 = np.random.default_rng(7)
for _ in range(reps):
    X = rng2.normal(size=(n, n_feat))
    yv = 1.0 + X @ np.array([0.5, -0.3, 0.8]) + rng2.normal(size=n)

    m_old = fit(X, yv)
    X_new = np.column_stack([X, rng2.normal(size=n)])   # 与 y 完全无关的噪声变量
    m_new = fit(X_new, yv)

    d = m_new.rsquared - m_old.rsquared
    da = m_new.rsquared_adj - m_old.rsquared_adj
    dR2.append(d)
    dadj.append(da)

    if d > 1e-12:
        R2_up += 1
    elif d < -1e-12:
        R2_down += 1
    else:
        R2_same += 1

    if da > 0:
        adj_up += 1

    # 判据验证: 调整 R^2 上升 <=> 新增变量的 |t| > 1
    t_new = abs(m_new.tvalues[-1])
    if (da > 0) != (t_new > 1):
        t_consistency = False

print(f"重复次数 = {reps},  样本量 n = {n},  原有特征数 p = {n_feat}")
print(f"\nR^2 的变化:")
print(f"  上升  的次数 = {R2_up:>6}  ({R2_up/reps*100:5.2f}%)")
print(f"  不变  的次数 = {R2_same:>6}  ({R2_same/reps*100:5.2f}%)")
print(f"  下降  的次数 = {R2_down:>6}  ({R2_down/reps*100:5.2f}%)")
print(f"  平均 ΔR^2    = {np.mean(dR2):.3e}   (恒 >= 0, 期望约为 sigma^2/TSS)")

print(f"\n调整 R^2 的变化:")
print(f"  上升  的次数 = {adj_up:>6}  ({adj_up/reps*100:5.2f}%)")
print(f"  下降  的次数 = {reps-adj_up:>6}  ({(reps-adj_up)/reps*100:5.2f}%)")
print(f"  平均 Δ调整R^2 = {np.mean(dadj):.3e}")
print(f"\n判据 'Δ调整R^2 > 0  <=>  |t_新| > 1' 在 {reps} 次模拟中始终成立: {t_consistency}")

print(f"""
理论解释:
  设 n 个样本, 原模型含 p 个自变量。
    R^2     = 1 - RSS / TSS                       (TSS 与自变量无关, 固定不变)
    调整R^2 = 1 - (RSS/(n-p-1)) / (TSS/(n-1))
            = 1 - (1-R^2) * (n-1)/(n-p-1)

  加入噪声变量后 OLS 的可行域变大, 目标函数 min RSS 只可能更小:
      RSS_new <= RSS_old  =>  R^2_new >= R^2_old      (数学上必然, 不会下降)
  每加入一个变量, 残差自由度 n-p-1 减少 1, 惩罚因子 (n-1)/(n-p-1) 变大。
  由于噪声变量与 y 无关, RSS 的下降幅度仅约为一个 sigma^2(期望上
  E[RSS_new]=(n-p-2)sigma^2, E[RSS_old]=(n-p-1)sigma^2), 恰好被损失掉的
  一个自由度抵消, 因此 (1-R^2)*(n-1)/(n-p-1) 的分子略降、分母的惩罚上升,
  调整 R^2 大概率下降。
  精确判据: 调整 R^2 上升  <=>  新增变量的 |t| > 1 (等价于偏 F > 1)。
  对纯噪声变量, P(|t| > 1) ≈ 0.32 < 0.5, 所以"下降"才是常态。
""")
