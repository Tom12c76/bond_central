import numpy as np
import pandas as pd
import scipy.interpolate as inter
import scipy.stats as stats

def get_calc(df_hist, df_funds, start1, win, sma):
	ptf_val = df_hist * df_funds['Quant']
	ptf_val['Ptf'] = ptf_val.sum(axis=1)

	logret = np.log(ptf_val / ptf_val.shift(1))

	cumret = logret[start1:].copy()
	cumret.iloc[0] = 0
	cumret = cumret.ffill()
	cumret = cumret.cumsum()
	cumret = np.exp(cumret) - 1

	cumret_xs = cumret.subtract(cumret['Ptf'], axis=0)

	rolret = (np.exp(logret.rolling(win).sum()) - 1).dropna(how='all')
	rolret_sma = rolret.rolling(sma).mean()

	rolvol = (logret.rolling(win).std() * np.sqrt(win)).dropna(how='all')
	rolrar = rolret / rolvol
	rolrar_sma = rolrar.rolling(sma).mean()

	rolret = rolret[start1:]
	rolret_sma = rolret_sma[start1:]

	rolret_sma_spl = None  # Initialize to None
	try:
		spl_app = []
		for isin in rolret_sma:
			x = rolret_sma[isin].dropna().index.values.astype('datetime64[D]').astype(int)
			y = rolret_sma[isin].dropna().values

			if len(x) > 3:  # Minimum data points
				spl = inter.UnivariateSpline(x, y, s=0.005)
				spl_app.append(pd.DataFrame(spl(x), index=rolret_sma[isin].dropna().index, columns=[isin]))
			else:
				print(f"Skipping spline (rolret_sma): ISIN {isin} has too few data points.")
				spl_app.append(pd.DataFrame(index=rolret_sma[isin].dropna().index, columns=[isin]))  # Empty DF

		rolret_sma_spl = pd.concat(spl_app, axis=1)
	except Exception as e:
		print(f"Error calculating rolret_sma_spl: {e}")
		rolret_sma_spl = rolret_sma.copy()
		rolret_sma_spl[:] = np.nan

	rolrar = rolrar[start1:]
	rolrar_sma = rolrar_sma[start1:]

	rolrar_sma_spl = None
	rolrar_sma_acc = None

	try:
		spl_app = []
		acc_app = []
		for isin in rolrar_sma:
			x = rolrar_sma[isin].dropna().index.values.astype('datetime64[D]').astype(int)
			y = rolrar_sma[isin].dropna().values

			if len(x) > 3: # Minimum Data Points
				spl = inter.UnivariateSpline(x, y, s=0.05)
				acc = spl.derivative()
				spl_app.append(pd.DataFrame(spl(x), index=rolrar_sma[isin].dropna().index, columns=[isin]))
				acc_app.append(pd.DataFrame(acc(x), index=rolrar_sma[isin].dropna().index, columns=[isin]))
			else:
				print(f"Skipping spline (rolrar_sma): ISIN {isin} has too few data points.")
				spl_app.append(pd.DataFrame(index=rolrar_sma[isin].dropna().index, columns=[isin]))
				acc_app.append(pd.DataFrame(index=rolrar_sma[isin].dropna().index, columns=[isin]))

		rolrar_sma_spl = pd.concat(spl_app, axis=1)
		rolrar_sma_acc = pd.concat(acc_app, axis=1)
	except Exception as e:
		print(f"Error calculating rolrar_sma_spl/acc: {e}")
		rolrar_sma_spl = rolrar_sma.copy()
		rolrar_sma_spl[:] = np.nan
		rolrar_sma_acc = rolrar_sma.copy()
		rolrar_sma_acc[:] = np.nan

	rarcdf = pd.DataFrame(stats.norm.cdf(rolrar), index=rolrar.index, columns=rolrar.columns)

	metrics = {
		'close': df_hist,
		'logret': logret,
		'cumret': cumret,
		'cumret_xs': cumret_xs,
		'rolret': rolret,
		'rolret_sma': rolret_sma,
		'rolvol': rolvol,
		'rolrar': rolrar,
		'rolrar_sma': rolrar_sma,
		'rolret_sma_spl': rolret_sma_spl,
		'rolrar_sma_spl': rolrar_sma_spl,
		'rolrar_sma_acc': rolrar_sma_acc,
		'rarcdf': rarcdf
	}

	melted_dfs = []
	for metric_name, df in metrics.items():
		if df is not None and not df.empty:
			df_reset = df.reset_index()
			melted_df = df_reset.melt(id_vars=['Date'], var_name='ISIN', value_name='Value')
			melted_df['Metric'] = metric_name
			melted_dfs.append(melted_df)
			del df, df_reset

	tall_df = pd.concat(melted_dfs, ignore_index=True)
	if not tall_df.empty:
		tall_df.set_index(['ISIN', 'Date'], inplace=True)
		tall_df = tall_df.pivot(columns='Metric', values='Value')
		tall_df.columns.name = None
		tall_df.reset_index(inplace=False)
		tall_df = tall_df[list(metrics.keys())]
	else:
		print("Warning: tall_df is empty.  Returning empty DataFrame.")
		tall_df = pd.DataFrame()

	return ptf_val, logret, cumret, cumret_xs, rolret, rolret_sma, rolvol, rolrar, rolrar_sma, rolret_sma_spl, rolrar_sma_spl, rolrar_sma_acc, rarcdf
