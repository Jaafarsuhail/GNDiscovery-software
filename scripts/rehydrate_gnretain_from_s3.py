#!/usr/bin/env python3
"""Recreate GNRetain's 5,452-row local reference tables from Le Goff Supplementary Table S3.

No source workbook or reconstructed records are distributed by this script. The
reader must obtain the source workbook from the original publication. The row
selection, a documented one-record label override and the algorithm are fixed.

Usage:
  python scripts/rehydrate_gnretain_from_s3.py --source-xlsx path/to/41598_2025_10208_MOESM4_ESM.xlsx --root .
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

DOC_NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
EXPECTED_PUBLIC_ROWS=6416
EXPECTED_REFERENCE_ROWS=5452
CORRECTED_LABELS={4208: 0}  # Published S3: Positive; frozen GNRetain reference: RN
DIAG_PROPS=['Molecular weight','RDKit cLogP','TPSA','H-bond donors','H-bond acceptors','Rotatable bonds','Fraction Csp3','Aromatic rings']
FOCUS_PROPS=['Acidic pKa','Basic pKa','cLogD7.4','cLogS','cLogP','Formal charge']


def parse_xlsx_table(path):
    """Read one-sheet S3 columns from OpenXML using only Python's standard library.

    Return rows keyed by the stable 1-based reference ID. Fails closed if the
    published spreadsheet layout is different or has omitted records.
    """
    with zipfile.ZipFile(path) as archive:
        shared=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            root=ET.fromstring(archive.read('xl/sharedStrings.xml'))
            for si in root.findall(DOC_NS+'si'):
                shared.append(''.join(node.text or '' for node in si.iter(DOC_NS+'t')))
        xml=archive.read('xl/worksheets/sheet1.xml')
    sheet=ET.fromstring(xml)
    actual={}
    for row in sheet.iter(DOC_NS+'row'):
        n=int(row.attrib.get('r',0))
        if n!=6 and not 7<=n<=6422: continue
        rowcells={}
        for cell in row.findall(DOC_NS+'c'):
            cellref=cell.attrib['r']
            col=''.join(ch for ch in cellref if ch.isalpha())
            if col not in {'A','F','P','Q','AN','AO','AP','AQ'}:continue
            v=cell.find(DOC_NS+'v')
            if cell.attrib.get('t')=='inlineStr':
                content=''.join(x.text or '' for x in cell.iter(DOC_NS+'t'))
            elif v is None:content=None
            elif cell.attrib.get('t')=='s':content=shared[int(v.text)]
            else:
                try:content=float(v.text)
                except (TypeError, ValueError):content=v.text
            rowcells[col]=content
        actual[n]=rowcells
    header=actual.get(6,{})
    if header.get('F')!='Classification' or header.get('AQ')!='SMILES':
        raise ValueError('Not Le Goff Supplementary Table S3: expected Classifications and SMILES at F6/AQ6')
    if len([r for r in actual if r>=7])!=EXPECTED_PUBLIC_ROWS:
        raise ValueError('Source S3 is incomplete or uses an unexpected row layout')
    return {i: actual[i+6] for i in range(1,EXPECTED_PUBLIC_ROWS+1)}


def ion_class(vals):
    charge=float(vals.get('Formal charge',0))
    acid=float(vals.get('Acidic pKa',np.nan))
    base=float(vals.get('Basic pKa',np.nan))
    if charge>0:return 'Formally cationic'
    if charge<0:return 'Formally anionic'
    aa=np.isfinite(acid) and acid<7.4
    bb=np.isfinite(base) and base>7.4
    if aa and bb:return 'Ampholytic tendency'
    if bb:return 'Cationic tendency'
    if aa:return 'Anionic tendency'
    return 'Predominantly neutral tendency'


def coords_from_meta(df, meta):
    props=meta['properties']
    arr=df.loc[:,props].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(arr).all():raise ValueError('PCA reference contains non-finite input descriptors')
    mean=np.array(meta['mean'],dtype=float)
    scale=np.array(meta['scale'],dtype=float)
    comps=np.array(meta['components'],dtype=float)
    result=((arr-mean)/scale)@comps.T
    return result[:,0],result[:,1]


def quantiles(df,columns):
    rows=[]
    for prop in columns:
        values=pd.to_numeric(df[prop],errors='coerce').dropna().to_numpy(dtype=float)
        q=np.quantile(values,[0,.01,.05,.25,.50,.75,.95,.99,1])
        rows.append({'property':prop,**dict(zip(['q00','q01','q05','q25','q50','q75','q95','q99','q100'],q)),'n':len(values)})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-xlsx',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path('.'),help='GNDiscovery repo root')
    parser.add_argument('--manifest',type=Path,default=None)
    parser.add_argument('--compare-to',type=Path,default=None,help='Optional full private runtime root for numeric/row parity checks')
    args=parser.parse_args()
    root=args.root.resolve()
    repo_scripts=Path(__file__).resolve().parents[1]
    manifest=args.manifest or next((p for p in (repo_scripts/'manifests/gnretain_s3_row_ids.txt', repo_scripts/'data_manifests/gnretain_s3_row_ids.txt') if p.is_file()), None)
    if manifest is None:raise FileNotFoundError('Missing gnretain_s3_row_ids.txt manifest')
    ids=[int(s.strip()) for s in manifest.read_text().splitlines() if s.strip() and not s.startswith('#')]
    if len(ids)!=EXPECTED_REFERENCE_ROWS or len(set(ids))!=EXPECTED_REFERENCE_ROWS or not all(1<=i<=6416 for i in ids):
        raise ValueError('Incorrect or modified GNRetain source-row manifest')
    records=parse_xlsx_table(args.source_xlsx)
    sys.path.insert(0,str(root/'GNRetain_Source_Feature_Engine'))
    from app.modules.smiles_standardization import standardize_smiles
    keep=[];diagnostics=[];focused=[]
    changes=[]
    for i in ids:
        rec=records[i]
        raw=rec.get('AQ')
        if not isinstance(raw,str) or not raw.strip():raise ValueError(f'Missing source SMILES for S3 row ID {i}')
        published=str(rec.get('F','')).strip().lower()
        if published not in ('positive','negative'):
            raise ValueError(f'Unexpected source class at S3 data row {i}: {published!r}')
        base=int(published=='positive')
        label=CORRECTED_LABELS.get(i,base)
        if label!=base:changes.append({'compound_id':i,'source_label':published,'gnretain_label':'RN'})
        smi=standardize_smiles(raw)
        mol=Chem.MolFromSmiles(smi)
        if mol is None:raise ValueError(f'Failed to parse standardized S3 compound {i}')
        charge=int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
        keep.append({'compound_id':i,'smiles':raw,'runtime_smiles':smi,'label_retention':label})
        diag={'compound_id':i,'runtime_smiles':smi,'endpoint_first':'gnretention_v05','label_first':float(label),
              'Molecular weight':Descriptors.MolWt(mol),'RDKit cLogP':Crippen.MolLogP(mol),
              'TPSA':rdMolDescriptors.CalcTPSA(mol),'H-bond donors':float(Lipinski.NumHDonors(mol)),
              'H-bond acceptors':float(Lipinski.NumHAcceptors(mol)),
              'Rotatable bonds':float(Lipinski.NumRotatableBonds(mol)),
              'Fraction Csp3':rdMolDescriptors.CalcFractionCSP3(mol),
              'Aromatic rings':float(rdMolDescriptors.CalcNumAromaticRings(mol))}
        # Frozen diagnostic display table has one stereochemical drawing difference
        # from the training-neighbour table. This does not affect model features.
        if i == 2178:
            if '[C@H]2O' not in diag['runtime_smiles']:
                raise ValueError('Unexpected structure at S3 row 2178; refusing display correction')
            diag['runtime_smiles'] = diag['runtime_smiles'].replace('[C@H]2O','[C@@H]2O')
        diagnostics.append(diag)
        foc={'compound_id':i,'runtime_smiles':smi,'endpoint_first':'gnretention_v05','label_first':int(label),
             'Acidic pKa':float(rec['AN']),'Basic pKa':float(rec['AO']),
             'cLogD7.4':float(rec['AP']),'cLogS':float(rec['Q']),'cLogP':float(rec['P']),
             'Formal charge':charge}
        # Historical focused-reference charge annotation differs from RDKit's formal
        # charge for this single compound; freeze the displayed reference value.
        if i == 3638: foc['Formal charge'] = 0
        foc['ionization_class']=ion_class(foc)
        focused.append(foc)
    if changes != [{'compound_id':4208,'source_label':'positive','gnretain_label':'RN'}]:
        raise ValueError(f'Unexpected label-curation delta from published S3: {changes}')
    relbase=Path('GNRetain221_Support_Assets')
    ad=root/relbase/'ad_reference';meta=root/relbase/'metadata'
    ad.mkdir(parents=True,exist_ok=True);meta.mkdir(parents=True,exist_ok=True)
    train=pd.DataFrame(keep);diag=pd.DataFrame(diagnostics);foc=pd.DataFrame(focused)
    diag_meta=json.loads((ad/'diagnostic_pca_metadata.json').read_text())
    foc_meta=json.loads((ad/'focused_pca_metadata.json').read_text())
    diag_pc1,diag_pc2=coords_from_meta(diag,diag_meta)
    foc_pc1,foc_pc2=coords_from_meta(foc,foc_meta)
    dc=diag[['compound_id','runtime_smiles','endpoint_first','label_first']].copy()
    dc['pc1']=diag_pc1;dc['pc2']=diag_pc2
    fc=foc[['compound_id','runtime_smiles','endpoint_first','label_first','ionization_class']].copy()
    fc['pc1']=foc_pc1;fc['pc2']=foc_pc2
    tables={
      relbase/'metadata/GNRetain221_training_rows.csv':train,
      relbase/'ad_reference/diagnostic_descriptor_reference.csv':diag,
      relbase/'ad_reference/focused_descriptor_reference.csv':foc,
      relbase/'ad_reference/diagnostic_pca_reference.csv':dc,
      relbase/'ad_reference/focused_pca_reference.csv':fc,
      relbase/'ad_reference/diagnostic_descriptor_quantiles.csv':quantiles(diag,DIAG_PROPS),
      relbase/'ad_reference/focused_descriptor_quantiles.csv':quantiles(foc,FOCUS_PROPS),
    }
    # Fail closed before writing if a reference comparison has missing/mismatched IDs or structures.
    comparisons={}
    if args.compare_to:
        original=args.compare_to.resolve()
        for rel,tbl in tables.items():
            other=pd.read_csv(original/rel)
            if len(other)!=len(tbl) or list(other.columns)!=list(tbl.columns):
                raise ValueError(f'Column or row mismatch: {rel}')
            eq={}
            for col in tbl.columns:
                a=tbl[col];b=other[col]
                if pd.api.types.is_numeric_dtype(b):
                    mask=np.isclose(pd.to_numeric(a),pd.to_numeric(b),rtol=0,atol=1e-6,equal_nan=True)
                    eq[col]=int(np.sum(~mask))
                    if eq[col] and eq[col]<5:
                        ix=np.where(~mask)[0];print('DIFF',rel,col,[(int(tbl.iloc[j].compound_id) if 'compound_id' in tbl else j, a.iloc[j],b.iloc[j]) for j in ix[:5]])
                else:
                    mm=(a.astype(str)!=b.astype(str))
                    eq[col]=int(mm.sum())
                    if eq[col] and eq[col]<5:
                        ix=np.where(mm)[0];print('DIFF',rel,col,[(int(tbl.iloc[j].compound_id) if 'compound_id' in tbl else j,a.iloc[j],b.iloc[j]) for j in ix[:5]])
            comparisons[str(rel)]={'nonmatching_cells':eq,'all_equal':not any(eq.values())}
        print(json.dumps({'reference_comparisons':comparisons},indent=2),flush=True)
        failed={k:v for k,v in comparisons.items() if not v['all_equal']}
        if failed:raise RuntimeError('Reconstructed references did not reproduce the private model assets; NOT writing')
    for rel,tbl in tables.items():
        path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
        tbl.to_csv(path,index=False)
    print(json.dumps({'status':'OK','retention_rows':len(train),'source_records':len(records),
                     'source_xlsx_sha256':hashlib.sha256(args.source_xlsx.read_bytes()).hexdigest(),
                     'one_label_correction':changes,'files_created':[str(p) for p in tables]},indent=2))

if __name__=='__main__':main()
