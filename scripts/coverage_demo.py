#!/usr/bin/env python3
"""Build and run a frozen foreground-coverage detector using explicit normal images and raw queries."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import faiss
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from src.coverage_detector import CoverageDetector,build_coverage_bank


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['build','predict'])
    for name in ['bank','data-root','dino-repo','weights']:parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--images',required=True,nargs='+',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--category')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--head',choices=['baseline','coverage'],default='coverage')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(4);cv2.setNumThreads(4);faiss.omp_set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='build':
        if not args.config or not args.category:parser.error('build requires --config and --category')
        result=build_coverage_bank(args.images,args.bank,args.data_root,args.dino_repo,args.weights,json.loads(args.config.read_text()),args.category,args.device)
    else:
        detector=CoverageDetector(args.bank,args.dino_repo,args.weights,args.device,args.head);result=[]
        for path in args.images:
            row=detector.predict(path,args.data_root);patch=row.pop('patch_map');row['image']=str(path)
            if args.output:
                args.output.mkdir(exist_ok=True,parents=True);file=args.output/f'{len(result)}_forward_map.npy';np.save(file,patch);row['forward_map_file']=str(file)
            result.append(row)
        if args.output:(args.output/'predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
