# python 2.7
import os
import sys
from functools import partial

chunksize = 1024
maxchunks = 40000

def splitfile(filename, directory, chunksize=chunksize, maxchunks=maxchunks):
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        for fname in os.listdir(directory):
            os.remove(os.path.join(directory, fname))
    chunknum = 0
    with open(filename, 'rb') as infile:
        for chunk in iter(partial(infile.read, chunksize*maxchunks), ''):
            ofilename = os.path.join(directory, ('chunk%04d'%(chunknum)))
            outfile = open(ofilename, 'wb')
            outfile.write(chunk)
            outfile.close()
            chunknum += 1

splitfile('checkpoints/frozen_inference_graph.pb', 'frozen_model_chunks')

