#!/usr/bin/env python
import sys, os, gzip
from urllib import urlretrieve

DIR      = 'https://ossci-datasets.s3.amazonaws.com/mnist/' #'http://yann.lecun.com/exdb/mnist/'
TARGETS  = ['train-images.idx3-ubyte', 't10k-images.idx3-ubyte', 
            'train-labels.idx1-ubyte', 't10k-labels.idx1-ubyte']

#for t in TARGETS:
#	url = "%s%s.gz" % (DIR, t)
#	url = url.replace('.idx', '-idx')
#	urlretrieve(url, (t + '.gz'))


#-Script Body ------------------------------------------------------------------
if __name__ == "__main__":

	if all(os.path.exists(t) for t in TARGETS):
		print("-- MNIST dataset found.")
	else:
		print('-- Dowloading MNIST data ...')

		def progressHook(count, blockSize, totalSize):
			p = min(count*blockSize / float(totalSize), 1)
			sys.stdout.write("\r-- | %-25s[ %3d%% ] " % (t, p*100))
			sys.stdout.flush()

		for t in TARGETS:

			url = "%s%s.gz" % (DIR, t)
			url = url.replace('.idx', '-idx')
			urlretrieve(url, (t + '.gz'), progressHook)	

			with gzip.open((t + '.gz'), 'rb') as archive:
				content = archive.read()

			with open(t, 'wb') as target:
				target.write(content)

			os.remove((t + '.gz'))
		print('Done!')

