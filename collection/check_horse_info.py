import re, sys
sys.path.insert(0, 'vps_setup/src')
from config import NETKEIBA_RACE_URL
from utils import fetch_url
from bs4 import BeautifulSoup

race_id="202412290211"
url=NETKEIBA_RACE_URL.format(race_id=race_id)
print(f"Fetching {url}")
resp=fetch_url(url)
print(f"Status {resp.status_code} len {len(resp.text)}")
soup=BeautifulSoup(resp.text, 'lxml')
# Find HorseList rows
rows=soup.find_all('tr', class_='HorseList')
print(f"Found {len(rows)} rows")
if rows:
    row=rows[0]
    print("Row HTML snippet:")
    print(str(row)[:2000])
    # Try to find Horse_Info
    td=row.find('td', class_='Horse_Info')
    print("\ntd.Horse_Info text:", repr(td.get_text().strip()) if td else "NOT FOUND")
    # Also check all tds
    for td in row.find_all('td'):
        print(f"td class={td.get('class')} text={repr(td.get_text().strip()[:100])}")
    # Also check for sex/age pattern
    text=row.get_text()
    print("\nFull row text:", repr(text[:500]))
    # Look for age pattern like 牡, 牝, セ with digit
    m=re.search(r'[牡牝セ]\d', text)
    print("age pattern found:", m.group() if m else "none")
    # Look for horse link
    a=row.find('a', href=re.compile(r'/horse/'))
    print("horse link", a.get('href') if a else "none", a.get_text().strip() if a else "")
