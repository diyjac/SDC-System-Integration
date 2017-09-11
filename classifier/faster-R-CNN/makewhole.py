# python 2.7
import os
import sys
from functools import partial

chunksize = 1024

def joinfiles(directory, filename, chunksize=chunksize):
    print "restoring:", filename, "from directory:", directory
    if os.path.exists(directory):
        if os.path.exists(filename):
            os.remove(filename)
        output = open(filename, 'wb')
        chunks = os.listdir(directory)
        chunks.sort()
        for fname in chunks:
            fpath = os.path.join(directory, fname)
            with open(fpath, 'rb') as fileobj:
                for chunk in iter(partial(fileobj.read, chunksize), ''):
                    output.write(chunk)
        output.close()

joinfiles('frozen_model_chunks', 'checkpoints/frozen_inference_graph.pb')
print "Done!"

