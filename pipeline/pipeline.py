"""
Script to chain all processing of test data and predict from saved model
"""

"""

To run this script type following command in Terminal (Mac) or Command Prompt (Windows):

! python pipeline.py PredictChurn --local-scheduler --test-data-file ../Data/test.csv
"""
import os, pickle
import pandas as pd
from xgboost import XGBClassifier 
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import luigi


class LoadTestData(luigi.Task):
	"""Read and load test data into pandas dataframe.
	Process features before feeding to model

	"""
	test_data_file = luigi.Parameter()
	output_file = luigi.Parameter(default="features.csv")

	def output(self):
		return luigi.LocalTarget(self.output_file)

	def run(self):

		# read test data file
		data_folder_path = '../Data'
		test_data_path = os.path.join(data_folder_path,'test.csv')
		df_test = pd.read_csv(test_data_path)
		df_test.drop(columns='id', inplace=True)

		# encode categorical features
		model_folder_path = '../model'
		categorical_variables = ['state','area_code','international_plan','voice_mail_plan']
		encoder_path = os.path.join(model_folder_path,'categorical_encoder.pkl')
		encoder_file = open(encoder_path,'rb')
		encoder = pickle.load(encoder_file)
		encoder_file.close()
		df_test.loc[:,categorical_variables] = encoder.transform(df_test[categorical_variables])

		# scale features
		scaling = MinMaxScaler()
		df_test = pd.DataFrame(data=scaling.fit_transform(df_test),
								columns=df_test.columns.to_list())

		df_test.to_csv(self.output().path, index=False)


class PredictChurn(luigi.Task):
	"""Reads the processed features and feeds into trained model


	"""

	test_data_file = luigi.Parameter()
	output_file = luigi.Parameter(default='prediction.csv')

	def requires(self):
		return LoadTestData(self.test_data_file)

	def output(self):
		return luigi.LocalTarget(self.output_file)

	def run(self):
		model_folder_path = '../model'
		model_path = os.path.join(model_folder_path,'xgb_final.pkl')
		trained_model = XGBClassifier()
		trained_model.load_model(model_path)

		# read the processed features file
		df_test = pd.read_csv(self.input().path)

		# predict churn
		prediction = trained_model.predict(df_test)

		# putting prediction in dataframe as well as index ids (this is the 
		#	the submission format)
		submission = pd.DataFrame(data=prediction, columns=['churn'])
		submission['churn'] = submission['churn'].map({1:'yes',
														0:'no'})
		submission.reset_index(inplace=True)
		submission.rename(columns={'index':'id'}, inplace=True)
		submission['id'] = submission['id'] + 1
		# write submission to file
		submission.to_csv(self.output().path, index=False)


if __name__ == "__main__":
	luigi.run()
