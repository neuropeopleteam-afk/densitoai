import os
import pandas as pd, pydicom, os, collections
lab = pd.read_csv('data/labels_for_embeddings.csv')
rows=[]
tags = ['Laterality','ImageLaterality','ViewPosition','PatientOrientation','BodyPartExamined','SeriesDescription','ImageComments','ProtocolName','StudyDescription','AcquisitionDeviceProcessingDescription','SeriesNumber','InstanceNumber','ImageType','Modality','ImageOrientationPatient','FieldOfViewRotation','FieldOfViewHorizontalFlip','PerformedProcedureStepDescription','AnatomicRegionSequence','ProcedureCodeSequence','ScheduledProcedureStepDescription','RequestedProcedureDescription','ContentDescription','ContentLabel']
allkw = collections.Counter()
for i,r in lab.iterrows():
    ds = pydicom.dcmread(r.file_path, force=True, stop_before_pixels=True)
    d = {'file_path': r.file_path, 'region': r.region, 'instance_number': r.instance_number, 'fname': os.path.basename(r.file_path), 'series_dir': os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(r.file_path))))}
    for t in tags:
        v = ds.get(t, None)
        d[t] = None if v is None else str(v.value if hasattr(v,'value') else v)
    for el in ds.iterall():
        allkw[(el.tag, el.keyword or el.name)] += 1
    # private tags
    priv = [(str(el.tag), str(el.value)[:60]) for el in ds if el.tag.is_private]
    d['n_private'] = len(priv)
    d['private_sample'] = ';'.join(f'{t}={v}' for t,v in priv[:12])
    rows.append(d)
df = pd.DataFrame(rows)
df.to_csv('out/dicom_tags_499.csv', index=False)
for t in tags:
    print(t, df[t].value_counts(dropna=False).head(8).to_dict())
print('series_dir', df.series_dir.value_counts().head(10).to_dict())
print('fname by region', df.groupby('region').fname.value_counts().head(20))
print('\nall keywords (count over files):')
for (tag,kw),c in sorted(allkw.items(), key=lambda x: str(x[0][0])):
    print(tag, kw, c)
