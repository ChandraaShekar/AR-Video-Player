# import the necessary packages
import numpy as np
import csv

from collections import Counter

class Searcher:
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath

	def search(self, queryFeatures, limit = 1):
		# initialize our dictionary of results
		results = {}

		# open the index file for reading
		with open(self.indexPath) as f:
			# initialize the CSV reader
			reader = csv.reader(f)

			# loop over the rows in the index
			for row in reader:
				# parse out the image ID and features, then compute the
				# chi-squared distance between the features in our index
				# and our query features
				features = [float(x) for x in row[1:]]
				d = self.chi2_distance(features, queryFeatures)

				# now that we have the distance between the two feature
				# vectors, we can udpate the results dictionary -- the
				# key is the current image ID in the index and the
				# value is the distance we just computed, representing
				# how 'similar' the image in the index is to our query
				results[row[0]] = d

			# close the reader
			f.close()

		# sort our results, so that the smaller distances (i.e. the
		# more relevant images are at the front of the list)
		results = sorted([(v, k) for (k, v) in results.items()])

		# return our (limited) results
		return results[:limit]

	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return d

	def k_nearest_neighbor(predict):
		k = 3
		# if len(data) >= k:
		# 	warnings.warn("K is set to a value less than total groups")

		k_distances = []

		# for group in data:
		# 	for n_features in data[group]:
		# 		eucledian_distance = np.linalg.norm(np.array(n_features) - np.array(predict))
		# 		k_distances.append([eucledian_distance, group])

		with open("index.csv") as f:
			data = csv.reader(f)
			for row in data:
				n_features = [float(i) for i in row[1:]]
				index = row[0]
				eucledian_distance = np.linalg.norm(np.array(n_features) - np.array(predict))
				k_distances.append([eucledian_distance, index])


		votes = [i[1] for i in sorted(k_distances)[:k]]

		vote_res = Counter(votes).most_common(1)[0][0]

		return vote_res