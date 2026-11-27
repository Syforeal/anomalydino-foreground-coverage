#!/usr/bin/env python3
"""Download the exact MVTec category archives recorded by this experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

PROJECT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=PROJECT/'results/data_manifest_all15.json')
    parser.add_argument('--category',nargs='+')
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--downloads',required=True,type=Path)
    args=parser.parse_args();args.root.mkdir(exist_ok=True,parents=True);args.downloads.mkdir(exist_ok=True,parents=True)
    records=json.loads(args.manifest.read_text())
    for record in records:
        if args.category and record['category'] not in args.category:continue
        archive=args.downloads/(record['category']+'.tar.xz')
        if not archive.exists():
            partial=archive.with_suffix('.partial')
            request=urllib.request.Request(record['url'],headers={'User-Agent':'AnomalyDINO-course-reproduction/1.0'})
            with urllib.request.urlopen(request,timeout=120) as response,partial.open('wb') as output:
                while block:=response.read(1<<20):output.write(block)
            if digest(partial)!=record['sha256']:raise ValueError('Archive hash mismatch')
            partial.replace(archive)
        if digest(archive)!=record['sha256']:raise ValueError('Archive hash mismatch')
        root=args.root.resolve()
        with tarfile.open(archive) as source:
            members=source.getmembers()
            for member in members:
                if not (root/member.name).resolve().is_relative_to(root) or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe archive member')
            complete=all((root/m.name).exists() and (not m.isfile() or (root/m.name).stat().st_size==m.size) for m in members)
            if not complete:source.extractall(root)
        print('READY',record['category'],flush=True)


if __name__=='__main__':main()
