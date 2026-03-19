import requests

login_url = 'https://www.ac.upc.edu/app/auth/login'
back_url = 'https://www.ac.upc.edu/app/wiki/serveis-tic/Clusters/Sert/FuncionamentGeneral'

combos = [
    ('gmoreno', 'Tallypass+0504'),
    ('gmoreno', 'Slenderman,2004'),
    ('guillem.moreno.garcia', 'Tallypass+0504'),
    ('guillem.moreno.garcia', 'Slenderman,2004'),
    ('gmoreno', 'Mamareentanga54'),
]

for user, pwd in combos:
    s = requests.Session()
    r = s.post(login_url,
               data={'username': user, 'password': pwd, 'back': back_url},
               allow_redirects=True, timeout=15)
    success = 'login' not in r.url.lower() and r.status_code == 200
    status = 'OK' if success else 'FAIL'
    print(user + ' / ' + pwd[:8] + '... -> ' + r.url[:70] + ' [' + status + ']')
    if success:
        print('SUCCESS - fetching wiki page...')
        wiki = s.get(back_url, timeout=15)
        print(wiki.text[:3000])
        break
