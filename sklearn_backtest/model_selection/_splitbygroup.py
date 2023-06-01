import numpy as np
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class TimeSeriesSplitByGroup:
	def __init__(
		self,
		n_splits=None,
		*,
		max_train_size=None,
		min_train_size=0,
		test_size=1,
		gap=0,
	):
		self.n_splits = n_splits
		self.max_train_size = max_train_size
		self.min_train_size = min_train_size
		self.test_size = test_size
		self.gap = gap

	def split(self, X, y=None, groups=None):

		if groups is None:
			raise ValueError("'groups' should not be None")
		
		if np.all(np.array(groups)[:-1] >= np.array(groups)[1:]):
			raise NotImplementedError("X, Y, groups should be sorted in groups ascending order")
		# TODO: deal with unordered groups.
		# indices=group_inverse?
		# group_starts = np.sort(group_starts)
		# unique_groups = groups[group_starts]



		# Ensure X, y, and groups are indexable
		X, y, groups = indexable(X, y, groups)
		n_samples = _num_samples(X)

		# all those arguments refer to GROUPS, not samples
		gap = self.gap
		test_size = self.test_size
		max_train_size = self.max_train_size
		min_train_size = self.min_train_size

		


		# Get unique groups
		unique_groups, group_starts = np.unique(groups, return_index=True)
		n_groups = _num_samples(unique_groups)
		group_starts = np.append(group_starts, n_samples)
		
		# Calculate n splits
		# |---n_unique_groups---| = |---train size---| + |---gap---| + |---test size * n_splits ---|
		n_splits = (n_groups - gap - min_train_size) // test_size
		
		if n_splits < 1:
			raise ValueError("n_splits < 1")

		if not hasattr(self, "_n_splits"):
			print("setting _n_splits")
			self.n_splits_ = n_splits
			self.unique_groups_ = unique_groups
			self.group_starts_ = group_starts

		

		indices = np.arange(n_samples) 
		test_starts = range(
			n_groups - (n_splits * test_size), n_groups, test_size
		)
		for te_start in test_starts:
			tr_end = te_start - gap
			tr_start = (tr_end - max_train_size) if max_train_size else 0
			idx_train_start, idx_train_end = (
				group_starts[tr_start],
				group_starts[tr_end],
			)

			idx_test_start, idx_test_end = (
				group_starts[te_start],
				group_starts[te_start + test_size],
			)
			
			yield indices[idx_train_start:idx_train_end], indices[idx_test_start:idx_test_end]


if __name__ == "__main__":
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt

	# Create a pandas DataFrame with date (monthly), x, y, and ticker columns for two different tickers
	n_samples = 15
	df = pd.DataFrame(
		dict(
			date=pd.period_range("2022-04", periods=5, freq="M").tolist()
			+ pd.period_range("2022-01", periods=10, freq="M").tolist(),
			x=np.random.randn(n_samples),
			y=np.random.randn(n_samples),
			tickers=["AAPL"] * 5 + ["GOOG"] * 10,
		)
	)

	df = df.sort_values("date", ascending=False).reset_index(drop=True)

	X = df
	y = df["y"]
	groups = df["date"].astype(str)

	cv_args = dict(
		test_size=2,
		gap=1,
		max_train_size=3,
		min_train_size=3,
		n_splits=None,
	)

	cv = TimeSeriesSplitByGroup(**cv_args)

	splits = list(cv.split(X, groups=groups))
	n_splits = cv.n_splits_

	mtx = pd.DataFrame("", index=groups, columns=range(n_splits))
	mtx = mtx.rename_axis(["split"], axis=1)

	for i, (tr_idx, te_idx) in enumerate(splits):
		mtx.iloc[tr_idx, i] = "O"
		mtx.iloc[te_idx, i] = "#"

	print(*[f"{k}={v}" for k, v in cv_args.items()])
	print(f"{cv.n_splits_=}")
	mtx
