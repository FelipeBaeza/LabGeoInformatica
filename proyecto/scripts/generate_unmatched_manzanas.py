#!/usr/bin/env python3
"""Compara un CSV/DBF de manzanas (INE) con el GeoJSON local y genera unmatched_manzanas.csv

Uso:
  python3 generate_unmatched_manzanas.py --csv path/to/ine_manzanas.csv

Salida:
  proyecto/outputs/unmatched_manzanas.csv
"""
import argparse
import csv
from pathlib import Path
import json

def load_geo_manzent(geojson_path):
    # lee el geojson y extrae los MANZENT (o MANZANA/CODIGO_DIS) claves
    vals=set()
    with open(geojson_path,'r',encoding='utf-8',errors='replace') as f:
        data=json.load(f)
        for feat in data.get('features',[]):
            props=feat.get('properties',{})
            # intenta MANZENT, MANZANA, CODIGO_DIS
            if 'MANZENT' in props:
                vals.add(str(props['MANZENT']))
            else:
                key=None
                if 'MANZANA' in props and 'CODIGO_DIS' in props:
                    vals.add(f"{props.get('CODIGO_DIS')}_{props.get('MANZANA')}")
                elif 'MANZANA' in props:
                    vals.add(str(props.get('MANZANA')))
    return vals


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='CSV INE con MANZENT o CODIGO_DIS+MANZANA')
    p.add_argument('--geo', default='Datos GeoJSON/manzana_metadato.geojson')
    p.add_argument('--out', default='proyecto/outputs/unmatched_manzanas.csv')
    args=p.parse_args()

    geo_vals=load_geo_manzent(args.geo)
    out_path=Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.csv,'r',encoding='utf-8',errors='replace') as cf, open(out_path,'w',newline='',encoding='utf-8') as of:
        reader=csv.DictReader(cf)
        fieldnames=reader.fieldnames+['matched_exact'] if 'matched_exact' not in reader.fieldnames else reader.fieldnames
        writer=csv.DictWriter(of, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            # check MANZENT first
            manzent=row.get('MANZENT') or row.get('manzent') or ''
            if manzent and str(manzent) in geo_vals:
                row['matched_exact']='yes'
            else:
                # try CODIGO_DIS + MANZANA
                cod=row.get('CODIGO_DIS') or row.get('codigo_dis') or row.get('CODIGO') or ''
                man=row.get('MANZANA') or row.get('manzana') or ''
                key=f"{cod}_{man}" if cod and man else ''
                if key and key in geo_vals:
                    row['matched_exact']='yes'
                else:
                    row['matched_exact']='no'
            writer.writerow(row)
    print(f'Wrote unmatched report to {out_path}')

if __name__=='__main__':
    main()
