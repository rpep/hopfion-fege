import os
import logging
import sys

try:
    sys.path.append('/io/finmag/src')
    import finmag
except:
    jid = ''
    aid = ''
    if 'SLURM_JOB_ID' in os.environ.keys():
        jid = os.environ['SLURM_JOB_ID']
    if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
        aid = os.environ['SLURM_ARRAY_TASK_ID']

    path = '/tmp/finmag-{}-{}/src'.format(jid, aid)
    print('Trying Finmag path: {}'.format(path))
    sys.path.append(path)

import finmag
