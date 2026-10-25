"""Prepare the completed Boiler release for a fresh reconstruction run.

Download xinan_completed_data.csv from the provider before using this helper.
Use --metadata-column only for a separate timestamp or exported row identifier.
All 30 measurement columns are retained in their supplied order.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('external/boiler_reconstruction'))
    parser.add_argument('--metadata-column',action='append',default=[],
                        help='Exact heading of a timestamp or exported row-ID column, if present')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    frame=pd.read_csv(args.file)
    measurements=frame.drop(columns=args.metadata_column)
    if measurements.shape!=(86400,30):
        raise ValueError('Expected 86400 rows and 30 measurement columns. '
                         'If the source also has a timestamp or row ID, name it with --metadata-column. '
                         'Retain every measurement column and every source row.')
    values=measurements.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('The completed release must have finite measurements; inspect the source file before proceeding.')
    names=list(map(str,measurements.columns))
    if 'row_index' in names:
        raise ValueError('The output ordering name row_index conflicts with a source measurement heading.')
    catalog=json.loads((args.root/'reproduction/run_catalog.json').read_text(encoding='utf-8'))
    template=next(r for r in catalog if r['id']=='relu_ETTh1_iTransformer_h96_s2026')
    saved=(args.root/template['record']).read_bytes()
    if hashlib.sha256(saved).hexdigest()!=template['record_sha256']:
        raise ValueError('The configuration template differs from the recorded release')
    cfg=copy.deepcopy(json.loads(saved)['config'])
    target=args.output/'boiler_completed.csv'
    if target.exists():
        raise FileExistsError('Choose a new output directory to preserve the existing CSV')
    args.output.mkdir(parents=True,exist_ok=True)
    out=measurements.copy()
    out.insert(0,'row_index',np.arange(len(out)))
    out.to_csv(target,index=False,float_format='%.17g')
    train_end=int(len(out)*.7);val_end=train_end+int(len(out)*.1)
    cfg['data'].update(path=str(target.resolve()),timestamp_column='row_index',length=len(out),channels=30)
    cfg['output_dir']=str((args.output/'run_boiler_relu_iTransformer_h96_s2026').resolve())
    config_path=args.output/'boiler_relu_iTransformer_h96_s2026.json'
    config_path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    record={
        'purpose':'Specified input protocol for a fresh industrial run of the released reconstruction.',
        'relation_to_original_results':'This protocol is specified for the released reconstruction; it is not a recovered original input manifest or a rerun of the original table.',
        'source_doi':'https://doi.org/10.6084/m9.figshare.28868849',
        'specified_source_version':'xinan_completed_data.csv',
        'supplied_filename':args.file.name,'source_sha256':hashlib.sha256(args.file.read_bytes()).hexdigest(),
        'measurement_shape':list(values.shape),'measurement_columns':names,
        'excluded_source_metadata_columns':args.metadata_column,
        'source_rows_zero_based_half_open':[0,len(out)],
        'output_file':target.name,'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'ordering_column':'row_index','ordering_column_meaning':'Zero-based source row ID, excluded from model inputs and targets.',
        'missing_values':0,'additional_imputation':'None; use the provider-completed values.',
        'resampling':'None; supplied row order is preserved.',
        'split_rows_zero_based_half_open':{'train':[0,train_end],'validation':[train_end,val_end],'test':[val_end,len(out)]},
        'standardization':'Training runner fits per-variable mean and population standard deviation on training rows; scales below 1e-6 are replaced by 1.',
        'windows':{'history':96,'target_lengths':[96,192,336,720],
                   'assignment':'Each target lies wholly within its partition; preceding history may cross the boundary.'},
        'runnable_example':{'config_file':config_path.name,'template_id':template['id'],
            'template_record':template['record'],'template_record_sha256':template['record_sha256'],
            'prediction_length':96,'seed':2026,
            'modified_fields':['data.path','data.timestamp_column','data.length','data.channels','output_dir'],
            'other_settings':'Copied exactly from the recorded ReLU/iTransformer configuration.'},
        'training_started':False}
    (args.output/'input_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({'csv':str(target),'measurement_shape':list(values.shape),
        'split_rows':record['split_rows_zero_based_half_open'],'training_started':False},indent=2))


if __name__=='__main__':main()
