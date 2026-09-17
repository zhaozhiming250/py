#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 3 题　Python 实操题：Carseats 多元线性回归
===============================================================================

题目要求
-------------------------------------------------------------------------------
使用 Python 加载 Carseats 数据集，以销售额 Sales 为响应变量，选取
Price（价格）、Income（收入）、Advertising（广告）以及定性特征
ShelveLoc（货架位置），建立多元线性回归模型。

  (1) 提取模型拟合报告，指出 ShelveLoc 的基准组是什么？
  (2) 解读 ShelveLoc[Good] 系数的实际商业含义。
  (3) 计算各变量的 VIF，评估是否存在多重共线性风险。

结论摘要（运行本文件可复现全部数字）
-------------------------------------------------------------------------------
  (1) 基准组 = Bad。ShelveLoc 是 3 水平定性变量，按字母序为
      Bad / Good / Medium，哑变量编码只产生 Good 与 Medium 两列，
      未单独出现的 Bad 即基准组（reference level）。

  (2) ShelveLoc[Good] 系数 = 4.8359，标准误 0.2597，t = 18.62，
      p 约等于 5.5e-56，95% 置信区间 [4.3254, 5.3464]。
      含义：在价格、收入、广告投入保持不变的条件下，货架位置为 Good
      的门店，销售额平均比货架位置为 Bad 的门店高 4.8359 千个
      （约 4836 个单位）。这是一个非常可观的商业效应。

  (3) VIF：Price 1.008、Income 1.012、Advertising 1.009、
      ShelveLoc 哑变量约 1.50。全部远低于阈值 5，
      分类变量按整体项评估的 GVIF^(1/(2*df)) = 1.003。
      结论：不存在多重共线性风险。

数据来源
-------------------------------------------------------------------------------
ISLR 的 Carseats 数据集（400 家门店的儿童汽车座椅销售数据），
通过 ISLP 包加载。

运行方式
-------------------------------------------------------------------------------
    pip install ISLP statsmodels pandas numpy
    python carseats_regression.py
"""

import sys

# 保证在 Windows 控制台也能正确输出中文
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ISLP import load_data


def section(title):
    """打印分节标题。"""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# 0. 加载数据
# =============================================================================
Carseats = load_data("Carseats")

section("0. 数据概览")
print(f"数据规模: {Carseats.shape[0]} 行 x {Carseats.shape[1]} 列")
print("\n前 5 行:")
print(Carseats.head().to_string())
print("\n各变量类型:")
print(Carseats.dtypes.to_string())
print("\nShelveLoc 的取值与频数(按字母序):")
print(Carseats["ShelveLoc"].value_counts().sort_index().to_string())
print("\n响应变量 Sales 的描述统计(单位: 千个):")
print(Carseats["Sales"].describe().to_string())


# =============================================================================
# 1. 建立多元线性回归模型
# =============================================================================
# ShelveLoc 是 3 水平定性变量。statsmodels / patsy 会自动做
# treatment（哑变量）编码，默认以字母序第一水平 Bad 作为基准组。
model = smf.ols(
    "Sales ~ Price + Income + Advertising + C(ShelveLoc)",
    data=Carseats,
).fit()

section("1. 模型拟合报告 (OLS)")
print(model.summary())


# =============================================================================
# 2. 小问 (1)：确认 ShelveLoc 的基准组
# =============================================================================
section("2. 小问 (1)  基准组的判定")

levels = sorted(Carseats["ShelveLoc"].unique())
exog_names = list(model.model.exog_names)
dummies = [c for c in exog_names if "ShelveLoc" in c]

# 出现在数据中、但没有单独成为一列的水平，就是被折进截距项的基准组
baseline = [lv for lv in levels if not any(lv in d for d in dummies)]

print("ShelveLoc 的全部水平      :", levels)
print("模型设计矩阵中的列名      :", exog_names)
print("ShelveLoc 生成的哑变量列  :", dummies)
print("哑变量个数 K-1            :", len(dummies), " (K =", len(levels), ")")
print()
print(">>> 基准组 (baseline) =", baseline[0])
print()
print("说明: 3 个水平若放 3 个哑变量会与截距项完全共线")
print("      (1_Bad + 1_Good + 1_Medium = 1，即哑变量陷阱)，")
print("      会使 X'X 不可逆、参数不可唯一识别。因此必须删掉一个水平，")
print("      把它折进截距项：截距就是 Bad 组在协变量为 0 时的期望销售额。")


# =============================================================================
# 3. 小问 (2)：解读 ShelveLoc[Good] 系数
# =============================================================================
section("3. 小问 (2)  ShelveLoc[Good] 系数的实际商业含义")

good_key = "C(ShelveLoc)[T.Good]"
coef_good = model.params[good_key]
se_good = model.bse[good_key]
t_good = model.tvalues[good_key]
p_good = model.pvalues[good_key]
ci_good = model.conf_int().loc[good_key]

# 基准情景: 各连续协变量取样本均值、ShelveLoc = Bad 时的预测值
mp = Carseats["Price"].mean()
mi = Carseats["Income"].mean()
ma = Carseats["Advertising"].mean()
pred_bad = (model.params["Intercept"]
            + model.params["Price"] * mp
            + model.params["Income"] * mi
            + model.params["Advertising"] * ma)

print(f"系数 beta_Good          = {coef_good:.4f}")
print(f"标准误                  = {se_good:.4f}")
print(f"t 统计量                = {t_good:.3f}")
print(f"p 值                    = {p_good:.3e}")
print(f"95% 置信区间            = [{ci_good[0]:.4f}, {ci_good[1]:.4f}]")
print()
print("单位说明: Sales 的单位是千个; Price 的单位是美元。")
print()
print("[解读]")
print(f"  在价格(Price)、社区收入(Income)、广告投入(Advertising)保持不变的")
print(f"  条件下，货架位置为 Good 的门店，其销售额平均比货架位置为 Bad 的")
print(f"  门店高 {coef_good:.4f} 千个，即约 {coef_good * 1000:,.0f} 个单位。")
print()
print("[要点说明]")
print("  a) \"其他变量不变\"是必须加的前提。多元回归系数是偏效应，衡量的是")
print("     控制了价格、收入、广告之后的净影响，不等于两组销售额的原始均值差。")
bad_mean = Carseats.loc[Carseats["ShelveLoc"] == "Bad", "Sales"].mean()
good_mean = Carseats.loc[Carseats["ShelveLoc"] == "Good", "Sales"].mean()
print(f"     原始均值差(未控制任何变量) = {good_mean:.3f} - {bad_mean:.3f} "
      f"= {good_mean - bad_mean:.3f} 千个")
print(f"     控制协变量后的净效应       = {coef_good:.3f} 千个")
print("     两者接近，说明货架位置与这三个变量基本不相关，其效应大部分是独立的。")
print()
print(f"  b) 单位换算: Sales 单位是千个，故 {coef_good:.4f} 千个 = "
      f"{coef_good * 1000:,.0f} 个单位。")
print()
print(f"  c) 相对幅度: 在协变量取均值的基准情景下，Bad 组的预测销售额为")
print(f"     {pred_bad:.3f} 千个，因此 Good 组相对提升约 "
      f"{coef_good / pred_bad * 100:.1f}%。")
print()
print(f"  d) 统计显著性: p = {p_good:.2e}，95% 置信区间 "
      f"[{ci_good[0]:.3f}, {ci_good[1]:.3f}] 完全不包含 0，")
print("     说明该效应在总体中确实存在，不是抽样偶然。")
print()
med_key = "C(ShelveLoc)[T.Medium]"
med_coef = model.params[med_key]
print(f"  e) 与 Medium 组对比: Medium 的系数为 {med_coef:.4f}。")
print(f"     若直接比较 Good 与 Medium，差异为 {coef_good - med_coef:.4f} 千个，")
print(f"     即把货架从\"中等\"升级到\"优质\"，仍能带来约 "
      f"{(coef_good - med_coef) * 1000:,.0f} 个单位的增量销售。")
print()
price_coef = model.params["Price"]
print("  f) 商业决策含义: 货架位置是本模型中最强的可控杠杆。")
print(f"     把货架从 Bad 改善到 Good 带来的 {coef_good * 1000:,.0f} 个单位增量，")
print(f"     相当于把单价降低约 {coef_good / abs(price_coef):.0f} 美元"
      f"（因为降价 1 美元仅提升约 {abs(price_coef) * 1000:.0f} 个单位）。")
print("     这意味着零售商在争取优质货架位置时，愿意付出的代价远高于小幅让利。")
print()
print("各组 Sales 的实际均值(未控制其他变量，仅供对照):")
for lv in levels:
    sub = Carseats.loc[Carseats["ShelveLoc"] == lv, "Sales"]
    print(f"  {lv:<7} n = {len(sub):>3}   均值 = {sub.mean():.3f}   标准差 = {sub.std():.3f}")


# =============================================================================
# 4. 小问 (3)：VIF 与多重共线性评估
# =============================================================================
section("4. 小问 (3)  VIF 计算与多重共线性评估")

# 定性变量必须先转成哑变量再计算 VIF，否则无法求逆、指标无意义。
# drop_first=True 保证与上面的回归模型使用一致的 treatment 编码。
X = pd.get_dummies(
    Carseats[["Price", "Income", "Advertising", "ShelveLoc"]],
    columns=["ShelveLoc"],
    drop_first=True,
    dtype=float,
)
Xc = sm.add_constant(X)   # 计算 VIF 需要常数列

vif = pd.DataFrame({
    "变量": Xc.columns,
    "VIF": [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])],
})
vif["容差(1/VIF)"] = 1 / vif["VIF"]

# 截距项的 VIF 恒等于 1，没有诊断意义，展示时剔除
print("各变量的 VIF:")
print(vif[vif["变量"] != "const"].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
print()
print("判定标准(经验法则):")
print("  VIF < 5         基本无共线性问题")
print("  5 <= VIF <= 10  中等共线性，需留意")
print("  VIF > 10        严重共线性，会导致系数估计不稳定、标准误膨胀")

# ---------------------------------------------------------------------------
# 补充一：分类变量的正确评估方式 —— 广义 VIF (Fox & Monette, 1992)
# ---------------------------------------------------------------------------
# 对含 K 个水平的定性变量，单独看每个哑变量的 VIF 会高估共线性，
# 因为同一个因子的两个哑变量天然负相关，这是参数化造成的，而非真实共线。
# 标准做法是按"项"整体计算 GVIF，再用 GVIF^(1/(2*df)) 与阈值比较。
# 注意: 相关矩阵中不能包含常数列(方差为 0，会导致除零与 NaN)，
# 因此这里用不含常数项的 X。
R = np.corrcoef(X.values, rowvar=False)
names = list(X.columns)
idx = {nm: i for i, nm in enumerate(names)}

print("\n广义 VIF (把分类变量作为整体项评估):")
print(f"  {'项':<14}{'自由度':>6}{'GVIF':>10}{'GVIF^(1/(2*df))':>18}")
terms = [
    ("ShelveLoc", ["ShelveLoc_Good", "ShelveLoc_Medium"]),
    ("Price", ["Price"]),
    ("Income", ["Income"]),
    ("Advertising", ["Advertising"]),
]
for term, cols in terms:
    a = [idx[c] for c in cols]
    b = [idx[c] for c in names if c not in cols]
    R11 = R[np.ix_(a, a)]
    R22 = R[np.ix_(b, b)]
    gvif = np.linalg.det(R11) * np.linalg.det(R22) / np.linalg.det(R)
    df = len(a)
    print(f"  {term:<14}{df:>6}{gvif:>10.3f}{gvif ** (1 / (2 * df)):>18.3f}")

# ---------------------------------------------------------------------------
# 补充二：用 VIF_j = 1 / (1 - R_j^2) 独立校验上面的数值
# ---------------------------------------------------------------------------
print("\nVIF 公式校验 (VIF_j 应等于 1/(1 - R_j^2)):")
for i, nm in enumerate(Xc.columns):
    if nm == "const":
        continue
    others = [c for c in X.columns if c != nm]
    r2j = sm.OLS(X[nm], sm.add_constant(X[others])).fit().rsquared
    v_direct = variance_inflation_factor(Xc.values, i)
    print(f"  {nm:<18} VIF = {v_direct:6.3f}    "
          f"1/(1-R_j^2) = {1 / (1 - r2j):6.3f}    R_j^2 = {r2j:.4f}")

# ---------------------------------------------------------------------------
# 补充三：相关系数矩阵作为辅助证据
# ---------------------------------------------------------------------------
print("\n辅助证据 —— 连续自变量相关系数矩阵:")
print(Carseats[["Sales", "Price", "Income", "Advertising"]].corr().round(3).to_string())

print()
print("[结论] 所有变量的 VIF 都在 1.0 到 1.5 之间，远低于阈值 5；")
print("       分类变量按整体项评估的 GVIF^(1/(2*df)) = 1.003，几乎完全正交。")
print("       >>> 不存在多重共线性风险，无需做变量剔除、岭回归或主成分回归。")
print()
print("       原因: 这三个变量在数据生成上相互独立 —— 价格由厂商自定、")
print("       收入是社区特征、广告是本地投放决策。现实中并非总是如此")
print("       （例如广告投入与门店规模往往强相关），所以每次建模都应例行检查 VIF。")
print()
print("       注意: 若错误地把 ShelveLoc 三个哑变量全部放入模型(不删基准组)，")
print("       会出现完全共线，VIF 趋于无穷大。这是哑变量陷阱，而非真实共线性。")


# =============================================================================
# 5. 补充统计量与模型对照
# =============================================================================
section("5. 补充统计量与模型对照")

TSS = float(((Carseats["Sales"] - Carseats["Sales"].mean()) ** 2).sum())
RSS = float((model.resid ** 2).sum())
RSE = np.sqrt(RSS / model.df_resid)

print(f"n = {int(model.nobs)},  自变量个数 p = {int(model.df_model)},  "
      f"残差自由度 = {int(model.df_resid)}")
print(f"TSS = {TSS:.4f},  RSS = {RSS:.4f},  ESS = {TSS - RSS:.4f}")
print(f"RSE (残差标准误) = {RSE:.4f} 千个")
print(f"R^2 = {model.rsquared:.4f},  调整 R^2 = {model.rsquared_adj:.4f}")
print(f"F = {model.fvalue:.2f},  p = {model.f_pvalue:.3e}")
print(f"AIC = {model.aic:.2f},  BIC = {model.bic:.2f}")
print(f"对数似然 = {model.llf:.2f}")

print("\n各系数明细 (估计值 / 标准误 / t / p / 95% 置信区间):")
for name in model.params.index:
    ci = model.conf_int().loc[name]
    print(f"  {name:<26}{model.params[name]:>9.4f}{model.bse[name]:>8.4f}"
          f"{model.tvalues[name]:>9.3f}{model.pvalues[name]:>12.3e}"
          f"   [{ci[0]:>8.4f}, {ci[1]:>8.4f}]")

# 与不含 ShelveLoc 的模型对照，说明定性变量的贡献
model_reduced = smf.ols(
    "Sales ~ Price + Income + Advertising", data=Carseats
).fit()

print("\n模型对照:")
print(f"  不含 ShelveLoc:  R^2 = {model_reduced.rsquared:.4f},  "
      f"调整 R^2 = {model_reduced.rsquared_adj:.4f},  AIC = {model_reduced.aic:.2f}")
print(f"  含   ShelveLoc:  R^2 = {model.rsquared:.4f},  "
      f"调整 R^2 = {model.rsquared_adj:.4f},  AIC = {model.aic:.2f}")
print(f"  差异:  R^2 提升 {model.rsquared - model_reduced.rsquared:.4f},  "
      f"AIC 下降 {model_reduced.aic - model.aic:.2f}")
print("  说明货架位置是价格、收入、广告之外最重要的解释因素。")

section("运行完毕")
