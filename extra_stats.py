"""补充统计量: RSE / F 检验 / 分类变量的广义 VIF (GVIF)"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from ISLP import load_data

Carseats = load_data("Carseats")
m = smf.ols("Sales ~ Price + Income + Advertising + C(ShelveLoc)", data=Carseats).fit()

n, p = int(m.nobs), int(m.df_model)
TSS = float(((Carseats.Sales - Carseats.Sales.mean()) ** 2).sum())
RSS = float((m.resid ** 2).sum())
RSE = np.sqrt(RSS / m.df_resid)

print("=== 补充统计量 ===")
print(f"n = {n}, p(自变量个数，不含截距) = {p}, 残差自由度 = {int(m.df_resid)}")
print(f"TSS = {TSS:.4f}, RSS = {RSS:.4f}, ESS = {TSS-RSS:.4f}")
print(f"RSE (残差标准误) = {RSE:.4f} 千个")
print(f"R^2 = {m.rsquared:.4f}, 调整R^2 = {m.rsquared_adj:.4f}")
print(f"F = {m.fvalue:.2f}, p = {m.f_pvalue:.3e}")
print(f"AIC = {m.aic:.2f}, BIC = {m.bic:.2f}")
print(f"对数似然 = {m.llf:.2f}")
print(f"各变量均值: Price={Carseats.Price.mean():.3f}, "
      f"Income={Carseats.Income.mean():.3f}, Advertising={Carseats.Advertising.mean():.3f}")
print(f"Sales 标准差 = {Carseats.Sales.std(ddof=1):.4f}")
print("\n各系数: 估计值 / 标准误 / t / p / 95%CI")
for name in m.params.index:
    ci = m.conf_int().loc[name]
    print(f"  {name:<26} {m.params[name]:>9.4f} {m.bse[name]:>7.4f} "
          f"{m.tvalues[name]:>8.3f} {m.pvalues[name]:>10.3e}  [{ci[0]:>8.4f}, {ci[1]:>8.4f}]")

# ------------------------- 广义方差膨胀因子 GVIF (Fox & Monette, 1992) -------
# 对含 K 个水平的分类变量, 单独看每个哑变量的 VIF 会低估其共线性,
# 标准做法是计算该"项"整体的 GVIF, 并用 GVIF^(1/(2*df)) 与阈值比较。
X = pd.get_dummies(Carseats[["Price", "Income", "Advertising", "ShelveLoc"]],
                   columns=["ShelveLoc"], drop_first=True, dtype=float)
R = np.corrcoef(X.values, rowvar=False)
names = list(X.columns)
idx = {nm: i for i, nm in enumerate(names)}
print("\n=== GVIF (分类变量作为整体项) ===")
for term, cols in [("ShelveLoc", ["ShelveLoc_Good", "ShelveLoc_Medium"]),
                   ("Price", ["Price"]), ("Income", ["Income"]),
                   ("Advertising", ["Advertising"])]:
    a = [idx[c] for c in cols]
    b = [idx[c] for c in names if c not in cols]
    R11 = R[np.ix_(a, a)]
    R22 = R[np.ix_(b, b)]
    gvif = np.linalg.det(R11) * np.linalg.det(R22) / np.linalg.det(R)
    df = len(a)
    print(f"  {term:<12} 自由度={df}  GVIF={gvif:7.3f}   "
          f"GVIF^(1/(2*df))={gvif ** (1 / (2 * df)):6.3f}"
          + ("   <-- 分类变量按整体项评估" if df > 1 else ""))

# 每个哑变量的普通 VIF 公式校验: VIF_j = 1/(1-R_j^2)
from statsmodels.stats.outliers_influence import variance_inflation_factor
Xc = sm.add_constant(X)
print("\n=== 普通 VIF 校验 ===")
for i, nm in enumerate(Xc.columns):
    if nm == "const":
        continue
    others = [c for c in Xc.columns if c != nm]
    r2j = sm.OLS(Xc[nm], sm.add_constant(Xc[others])).fit().rsquared
    v_direct = variance_inflation_factor(Xc.values, i)
    print(f"  {nm:<18} VIF={v_direct:6.3f}   1/(1-R_j^2)={1/(1-r2j):6.3f}   R_j^2={r2j:.4f}")
