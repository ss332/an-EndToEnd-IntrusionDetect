import urllib3

http = urllib3.PoolManager()
r = http.request('GET', 'https://www.mayiwxw.com/77_77376/36936343.html')

f=open('novel.txt', mode='wb')
f.write(r.data)
f.close()