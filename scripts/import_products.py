"""Import configured product stores and verify document IDs and search results."""
import argparse
import json
import os
import ssl
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import httpx

from product_data import ROOT, dump

DATA = ROOT / 'fixtures/products'


def request(client, method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    if response.status_code >= 400:
        # Do not include endpoints, credentials or proxy response bodies in logs.
        raise RuntimeError(f'{method} {path}: HTTP {response.status_code}')
    return response.json()


def import_es():
    name = os.getenv('ES_INDEX', 'poizon_products_v1')
    path = '/' + quote(name, safe='')
    headers, auth = {}, None
    if os.getenv('ES_API_KEY'):
        headers['Authorization'] = 'ApiKey ' + os.environ['ES_API_KEY']
    elif os.getenv('ES_USERNAME'):
        auth = (os.environ['ES_USERNAME'], os.getenv('ES_PASSWORD', ''))
    ca = os.getenv('ES_CA_CERT')
    verify = ssl.create_default_context(cafile=ca) if ca else True
    expected = json.loads((DATA / 'es.index.json').read_text())
    # Cluster owns topology; serverless rejects explicit shard/replica settings.
    expected['settings'].pop('number_of_shards', None)
    expected['settings'].pop('number_of_replicas', None)
    docs = [json.loads(s) for s in (DATA / 'products.jsonl').read_text().splitlines()]
    with httpx.Client(base_url=os.environ['ES_URL'].rstrip('/'), headers=headers,
                      auth=auth, verify=verify, timeout=60) as client:
        check = client.get(path)
        if check.status_code == 404:
            request(client, 'PUT', path, json=expected)
        elif check.status_code == 200:
            maps = request(client, 'GET', path + '/_mapping')
            if set(maps) != {name}:
                raise RuntimeError('ES target must be one concrete index, not a multi-index alias')
            actual = maps[name]['mappings'].get('properties', {})
            for field, spec in expected['mappings']['properties'].items():
                if any(actual.get(field, {}).get(k) != v for k, v in spec.items()):
                    raise RuntimeError(f'Incompatible ES field: {field}')
        else:
            raise RuntimeError(f'ES target HTTP {check.status_code}')
        # Generate action metadata from configured index, not the fixture index name.
        lines = []
        for doc in docs:
            lines.extend([json.dumps({'index': {'_index': name, '_id': doc['product_id']}}),
                          json.dumps(doc, ensure_ascii=False)])
        bulk = request(client, 'POST', '/_bulk?refresh=wait_for', content=('\n'.join(lines)+'\n').encode(),
                       headers={'Content-Type': 'application/x-ndjson'})
        if bulk.get('errors'):
            failures = [i['index'].get('status') for i in bulk['items'] if i['index'].get('error')]
            raise RuntimeError(f'ES bulk failures: {failures}')
        readback = request(client, 'POST', path+'/_mget', json={'ids': [d['product_id'] for d in docs]})
        stored = {d['_id']: d.get('_source') for d in readback['docs'] if d.get('found')}
        if any(stored.get(d['product_id']) != d for d in docs):
            raise RuntimeError('ES source readback mismatch')
        query = json.loads((DATA / 'es.search.json').read_text())
        result = request(client, 'POST', path+'/_search', json=query)
        if result.get('timed_out') or result.get('_shards', {}).get('failed'):
            raise RuntimeError('ES search incomplete')
        return {'index': name, 'imported': len(docs), 'verified_sources': len(stored),
                'count': request(client, 'GET', path+'/_count')['count'],
                'query': query['query']['multi_match']['query'],
                'hits': [{'id': h['_id'], 'score': h['_score'], 'title': h['_source']['title']}
                         for h in result['hits']['hits']]}


def import_qdrant():
    name = os.getenv('QDRANT_COLLECTION', 'poizon_products_v1')
    path = '/collections/' + quote(name, safe='')
    config = json.loads((DATA/'vectors/qdrant.collection.json').read_text())
    body = json.loads((DATA/'vectors/qdrant.upsert.json').read_text())
    headers = {'api-key': os.environ['QDRANT_API_KEY']} if os.getenv('QDRANT_API_KEY') else {}
    with httpx.Client(base_url=os.environ['QDRANT_URL'].rstrip('/'),headers=headers,timeout=60) as client:
        check = client.get(path)
        if check.status_code == 404:
            request(client,'PUT',path,json=config)
        elif check.status_code == 200:
            actual = check.json()['result']['config']['params']['vectors']
            if any(actual.get(k) != v for k,v in config['vectors'].items()):
                raise RuntimeError('Incompatible Qdrant vector schema')
        else:
            raise RuntimeError(f'Qdrant target HTTP {check.status_code}')
        request(client,'PUT',path+'/points?wait=true',json=body)
        ids = [p['id'] for p in body['points']]
        records = request(client,'POST',path+'/points',json={'ids':ids,'with_payload':True,'with_vector':True})['result']
        stored = {p['id']:p for p in records}
        for point in body['points']:
            actual = stored.get(point['id'])
            if not actual or actual['payload'] != point['payload'] or len(actual['vector']) != len(point['vector']):
                raise RuntimeError('Qdrant point readback mismatch')
            if max(abs(a-b) for a,b in zip(actual['vector'],point['vector'])) > 1e-5:
                raise RuntimeError('Qdrant vector readback mismatch')
        query = json.loads((DATA/'vectors/qdrant.query.json').read_text())
        found = request(client,'POST',path+'/points/query',json=query)['result']['points']
        exact = request(client,'POST',path+'/points/query',json={
            'query':body['points'][0]['vector'],'limit':1,'with_payload':False})['result']['points']
        if not exact or exact[0]['id'] != ids[0] or exact[0]['score'] < 0.99:
            raise RuntimeError('Qdrant self-vector query failed')
        return {'collection':name,'imported':len(ids),'verified_points':len(records),
                'count':request(client,'POST',path+'/points/count',json={'exact':True})['result']['count'],
                'query':'银色项链','threshold':0.8,'hits':[
                    {'id':p['id'],'score':p['score'],'title':p['payload']['title']} for p in found],
                'self_vector_match':exact[0]['id'],'self_vector_score':exact[0]['score']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service',choices=['es','qdrant','both'],default='both')
    args=parser.parse_args()
    report={'checked_at':datetime.now(timezone.utc).isoformat()}
    failed=False
    for name, fn in [('es',import_es),('qdrant',import_qdrant)]:
        if args.service not in ('both',name): continue
        try:
            report[name]={'status':'ok',**fn()}
            print(name, 'ok', 'imported', report[name]['imported'], 'query_hits', len(report[name]['hits']))
        except Exception as exc:
            failed=True
            error=str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__
            report[name]={'status':'failed','error':error}
            print(name,error)
    output=ROOT/'docs/evidence'/f'import-{args.service}.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    dump(output,report)
    if failed: sys.exit(1)


if __name__=='__main__':
    main()
