#!/usr/bin/env python3
"""Verify pinned public LOCO mirror; extract train/test PNG only, without QA."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import zipfile

EXPECTED='e2bbcb234fcb7478199dde32a861642eadade7c10ea762958bc7d762f3cd82e9'
SOURCE='https://huggingface.co/datasets/jiang-cc/MMAD/resolve/c4ed190dcb530f2f673ab293e575ad32054bb3cf/MVTec-LOCO.zip'
CATEGORIES={'breakfast_box','juice_bottle','pushpins','screw_bag','splicing_connectors'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--manifest',required=True,type=Path)
    args=parser.parse_args();h=hashlib.sha256()
    with args.archive.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    if h.hexdigest()!=EXPECTED:raise ValueError('Pinned archive SHA256 mismatch')
    counts=collections.Counter();files=[]
    with zipfile.ZipFile(args.archive) as archive:
        for member in archive.infolist():
            p=Path(member.filename)
            if len(p.parts)!=5 or p.parts[0]!='MVTec-LOCO' or p.parts[1] not in CATEGORIES or p.suffix!='.png':continue
            if '..' in p.parts:raise ValueError('Unsafe archive path')
            counts['/'.join(p.parts[1:4])]+=1
            if p.parts[2] not in {'test','train'}:continue
            target=args.data/Path(*p.parts[1:]);target.parent.mkdir(exist_ok=True,parents=True)
            content=archive.read(member);target.write_bytes(content)
            files.append({'path':str(target.relative_to(args.data)),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'zip_crc':member.CRC})
    result={'dataset':'MVTec LOCO AD, public MMAD mirror','source_url':SOURCE,'archive_sha256':EXPECTED,
            'counts':dict(sorted(counts.items())),'extracted':files,'license':'CC BY-NC-SA 4.0; Bergmann et al. IJCV 2022',
            'note':'Mirror LFS integrity verified; canonical MVTec archive hash unavailable. No QA, caption or validation extraction.'}
    args.manifest.parent.mkdir(exist_ok=True,parents=True);args.manifest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['counts'],indent=2))


if __name__=='__main__':main()
