import matplotlib.pyplot as plt

import pandas as pd

class Forecast():

	@staticmethod
	def plot_oil_gain_bars(case:pd.DataFrame,base:pd.DataFrame=None,/,count:int=20,filename:str=None) -> None:
		"""
		Plots the top oil gain wells by comparing case and base scenarios.

		Parameters:
		- case (pd.DataFrame): The case dataset with well data.
		- base (pd.DataFrame): The base dataset for comparison.
		- count (int, optional): Number of top wells to display. Default is 20.
		- filename (str, optional): If provided, saves the plot as a PNG file; otherwise, displays it.

		Returns:
		- None
		"""
		sm3_to_kton = 0.000862  # Convert sm3 to kton

		# Find common columns
		common_columns = case.columns.intersection(base.columns)

		# Extract last row values (excluding first and last columns)
		case_row = case.loc[:,common_columns].iloc[-1,:]

		# Calculate the difference if base is provided
		if base is not None:
			case_row = case_row-base.loc[:,common_columns].iloc[-1,:]

		# print(type(case_row),case_row)

		# Sort differences in descending order and select top `count` values
		top_wells = case_row.nlargest(count)

		# top_wells.to_excel(f"{filename}.xlsx")

		# Plotting
		plt.figure(figsize=(10, 6))
		plt.bar(top_wells.index,top_wells.values*sm3_to_kton, color='skyblue')

		# Labels and title
		plt.ylabel('Səmərə Neft, min ton', fontsize=12)
		plt.title(f'{count} ən çox təsir görən quyu', fontsize=14)

		# Rotate x-axis labels for better readability
		plt.xticks(rotation=45, ha='right', fontsize=10)

		plt.tight_layout()  # Adjust layout to fit labels

		# Show or save the plot
		if filename:
			plt.savefig(f'dropbox/{filename}.png', dpi=300, bbox_inches='tight')
		else:
			plt.show()

	@staticmethod
	def plot_oil_gain_curves(*cases:pd.DataFrame,base:pd.DataFrame=None,filename:str=None,labels:list=None) -> None:
		"""
		Plots oil gain for multiple cases. If `base` is provided, it plots the difference from `base`,
		otherwise, it plots the raw values.

		Parameters:
		- base (pd.DataFrame, optional): The base dataset for comparison. If None, raw values are plotted.
		- *cases (pd.DataFrame): One or more case datasets.
		- filename (str, optional): If provided, saves the plot as a PNG file; otherwise, displays it.
		- labels (list, optional): Labels for each case dataset. Must match the number of cases.

		Returns:
		- None
		"""
		sm3_to_kton = 0.000862  # Convert sm3 to kton

		# Validate input cases
		if not cases:
			raise ValueError("At least one case dataset must be provided.")

		if labels and len(labels) != len(cases):
			raise ValueError("Number of labels must match the number of cases.")

		plt.figure(figsize=(10, 6))  # Set figure size

		# Plot each case scenario
		for index, case in enumerate(cases):
			label = labels[index] if labels else f"Case {index + 1}"
			if base is not None:
				y_values = (case.iloc[:, -1] - base.iloc[:, -1]) * sm3_to_kton  # Difference from base
			else:
				y_values = case.iloc[:, -1] * sm3_to_kton  # Raw values
			
			plt.plot(case.iloc[:, 0], y_values, label=label)

		# Labels, title, and formatting
		plt.ylabel('Səmərə Neft, min ton', fontsize=12)
		plt.title('Oil Gain Comparison' if base is not None else 'Oil Production Trends', fontsize=14)
		plt.grid(True, linestyle='--', alpha=0.6)  # Improve grid visibility
		plt.legend()

		# Rotate x-axis labels for better readability
		plt.xticks(rotation=45, ha='right', fontsize=10)

		plt.tight_layout()  # Adjust layout to fit labels

		# Show or save the plot
		if filename:
			plt.savefig(f'dropbox/{filename}.png', dpi=300, bbox_inches='tight')
		else:
			plt.show()

	@staticmethod
	def plot_injection_curves(*cases:pd.DataFrame,filename:str=None):

		pass

	@staticmethod
	def plot_pressure_curves(base:pd.DataFrame,filename:str=None,**cases:pd.DataFrame): #base_index=1,case_index=1,base_name=None,labels=None
		"""
		Plots pressure curves for a base dataset and additional cases.

		Parameters:
		- base (pd.DataFrame): The base dataset containing pressure data.
		- filename (str, optional): If provided, saves the plot as a PNG file; otherwise, displays it.
		- **cases (pd.DataFrame): Additional case datasets to plot, with keys as labels.

		Returns:
		- None
		"""
		bars_to_atm = 0.986923 # Conversion factor from bars to atm

		base_pressure = base.iloc[:,1]*bars_to_atm

		plt.figure(figsize=(10, 6))  # Set the figure size

		# Plot base pressure curve
		plt.plot(base.iloc[:,0],base_pressure,linestyle='-',color='black',label=base_label)
		# plt.plot(base.iloc[:,base_index-1],base_pressure,linestyle='-',color='black',label=base_name)

		# Plot additional cases
		for label,case in cases.items():
			plt.plot(case.iloc[:,0],case.iloc[:,1]*bars_to_atm,linestyle='--',label=label)
		# for index,case in enumerate(args):
		# 	plt.plot(case.iloc[:,case_index-1],case.iloc[:,case_index]*bars_to_atm,linestyle='--',label=labels[index])

		# Labels and formatting
		plt.ylabel('Təzyiq, atm', fontsize=12)
		plt.legend()
		plt.grid(True, linestyle='--', alpha=0.6)  # Improve grid visibility
		plt.xticks(rotation=45, ha='right', fontsize=10)

		plt.tight_layout()  # Ensure layout fits well

		# Show or save the plot
		if filename:
			plt.savefig(f'dropbox/{filename}.png',dpi=300,bbox_inches='tight')
		else:
			plt.show()