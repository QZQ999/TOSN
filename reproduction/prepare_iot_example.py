"""Prepare the verified public IoT example and record its exact input protocol.

This protocol defines a fresh industrial run using the released reconstruction.
The original industrial result table retains its own experimental provenance.
"""
from pathlib import Path
import argparse
import hashlib
import json
import copy
import numpy as np

EXPECTED_SHA256='8af624b260fa4f4b6044ea08073b1023458e33127ed04591c3d8ad6cb8e73ee6'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('external/industrial_examples'))
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1],
                        help='Root of the released code repository')
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
    catalogue=json.loads((args.root/'reproduction/run_catalog.json').read_text(encoding='utf-8'))
    template=next(r for r in catalogue if r['id']=='relu_ETTh1_iTransformer_h96_s2026')
    template_bytes=(args.root/template['record']).read_bytes()
    if hashlib.sha256(template_bytes).hexdigest()!=template['record_sha256']:
        raise ValueError('The configuration template differs from the recorded release')
    cfg=copy.deepcopy(json.loads(template_bytes)['config'])
    # The scientific CSV loader removes one ordering column before conversion.
    # Ordinal row IDs preserve ordering without inventing calendar timestamps.
    indexed=np.column_stack((np.arange(len(values)),values))
    np.savetxt(target,indexed,delimiter=',',header=','.join(['row_index',*columns]),
               comments='',fmt=['%d',*(['%.17g']*18)])
    n=len(values);train_end=int(n*.7);val_end=train_end+int(n*.1)
    cfg['data'].update(path=str(target.resolve()),timestamp_column='row_index',length=n,channels=18)
    cfg['output_dir']=str((args.output/'run_iot_relu_iTransformer_h96_s2026').resolve())
    config_path=args.output/'iot_relu_iTransformer_h96_s2026.json'
    config_path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    record={
        'purpose':'Specified input protocol for a fresh industrial run of the released reconstruction.',
        'relation_to_original_results':'This source selection is specified for the released reconstruction; it is not a recovered original input manifest or a rerun of the original table.',
        'source_archive':'UTSD-full-npy.zip','source_member':'UTSD-full-npy/IoT/baian/0.npy',
        'source_sha256':digest,'source_shape':list(values.shape),'source_dtype':str(values.dtype),
        'output_file':target.name,'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'measurement_columns':columns,'column_meaning':'Ordinal identifiers preserving the supplied column order.',
        'ordering_column':'row_index','ordering_column_meaning':'Zero-based source row ID, excluded from model inputs and targets.',
        'source_rows_zero_based_half_open':[0,n],
        'missing_values':0,'imputation':'None; source values are finite.',
        'resampling':'None; supplied row order is preserved.',
        'split_rows_zero_based_half_open':{'train':[0,train_end],'validation':[train_end,val_end],'test':[val_end,n]},
        'standardization':'The training runner fits per-column mean and population standard deviation on training rows. A standard deviation below 1e-6 is replaced by 1.',
        'windows':{'history':96,'target_lengths':[96,192,336,720],
                   'assignment':'Each target lies wholly within its partition; preceding history may cross the boundary.'},
        'runnable_example':{'config_file':config_path.name,'template_id':template['id'],
            'template_record':template['record'],'template_record_sha256':template['record_sha256'],
            'prediction_length':96,'seed':2026,
            'modified_fields':['data.path','data.timestamp_column','data.length','data.channels','output_dir'],
            'other_settings':'Copied exactly from the recorded ReLU/iTransformer configuration.'},
        'training_started':False}
    (args.output/'input_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({'csv':str(target),'rows':n,'columns':18,'split_rows':record['split_rows_zero_based_half_open'],'training_started':False},indent=2))

if __name__=='__main__':main()
