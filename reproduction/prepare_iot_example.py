"""Prepare the verified public IoT example and record its exact input protocol.

This source example is documented for new runs. Its selection is separate from
the unresolved file selection behind the original industrial result table.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

EXPECTED_SHA256='8af624b260fa4f4b6044ea08073b1023458e33127ed04591c3d8ad6cb8e73ee6'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('external/industrial_examples'))
    args=parser.parse_args()
    digest=hashlib.sha256(args.file.read_bytes()).hexdigest()
    if digest!=EXPECTED_SHA256:
        raise ValueError('Expected the verified UTSD-full-npy/IoT/baian/0.npy file')
    values=np.load(args.file,allow_pickle=False,mmap_mode='r')
    if values.shape!=(180140,18) or not np.isfinite(values).all():
        raise ValueError('Unexpected shape or missing values in the source example')
    args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/'iot_0.csv'
    columns=[f'v{i+1:02d}' for i in range(18)]
    if target.exists():
        raise FileExistsError('Choose a new output directory to preserve the existing CSV')
    np.savetxt(target,values,delimiter=',',header=','.join(columns),comments='',fmt='%.17g')
    n=len(values);train_end=int(n*.7);val_end=train_end+int(n*.1)
    record={
        'purpose':'Public source example for new runs; original industrial-table file selection remains unconfirmed.',
        'source_archive':'UTSD-full-npy.zip','source_member':'UTSD-full-npy/IoT/baian/0.npy',
        'source_sha256':digest,'source_shape':list(values.shape),'source_dtype':str(values.dtype),
        'output_file':target.name,'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'columns':columns,'column_meaning':'Ordinal identifiers preserving the supplied column order.',
        'source_rows_zero_based_half_open':[0,n],
        'missing_values':0,'imputation':'None; source values are finite.',
        'resampling':'None; supplied row order is preserved.',
        'split_rows_zero_based_half_open':{'train':[0,train_end],'validation':[train_end,val_end],'test':[val_end,n]},
        'standardization':'The training runner fits per-column mean and population standard deviation on training rows. A standard deviation below 1e-6 is replaced by 1.',
        'windows':{'history':96,'target_lengths':[96,192,336,720],
                   'assignment':'Each target lies wholly within its partition; preceding history may cross the boundary.'},
        'training_started':False}
    (args.output/'input_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({'csv':str(target),'rows':n,'columns':18,'split_rows':record['split_rows_zero_based_half_open'],'training_started':False},indent=2))

if __name__=='__main__':main()
