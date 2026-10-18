import urllib.request,json,hashlib,pathlib,datetime
r=pathlib.Path(__file__).resolve().parents[1];rows=[];sources=[]
for season in ['2021-22','2022-23','2023-24','2024-25']:
 url=f'https://raw.githubusercontent.com/openfootball/football.json/master/{season}/en.1.json';raw=urllib.request.urlopen(url).read();data=json.loads(raw)
 valid=[dict(season=season,date=m['date'],home=m['team1'],away=m['team2'],hg=m['score']['ft'][0],ag=m['score']['ft'][1])for m in data['matches']if len(m.get('score',{}).get('ft',[]))==2 and all(isinstance(n,int)for n in m['score']['ft'])]
 assert len(valid)==380,(season,len(valid));rows+=valid;sources.append(dict(season=season,url=url,sha256=hashlib.sha256(raw).hexdigest(),rows=len(valid)))
rows.sort(key=lambda m:(m['date'],m['home']));(r/'src/data/football.json').write_text(json.dumps(rows,separators=(',',':')))
(r/'src/data/football-provenance.json').write_text(json.dumps(dict(source='openfootball/football.json',license='CC0-1.0',retrievedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=len(rows),sources=sources,scope='Historical Premier League results. No odds, injuries, lineups, player data or live match feed.'),indent=2))
license=urllib.request.urlopen('https://raw.githubusercontent.com/openfootball/football.json/master/LICENSE.md').read();(r/'public/FOOTBALL-LICENSE.md').write_bytes(license)
print(f'Fetched {len(rows)} completed matches across 4 seasons')
