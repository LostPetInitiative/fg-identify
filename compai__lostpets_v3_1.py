from __future__ import print_function, absolute_import, division
import argparse
import sys
import gzip
from time import time

description = """Make a submission to the challenge. Examples:
{executable} submit FILE
{executable} submit FILE --submit_comment 'Tried another approach ...'
""".format(executable=sys.argv[0])

try:
    import requests
except:
    print('Please install "requests" package first (pip install requests)')

server = 'http://92.63.96.33/'
auth_options = dict(
    challenge_name='_lostpets_v3_1',
    user_email= 'evilpsycho42@gmail.com',
    user_token='247fb599bc047ce85cb06bd95203764b',
)
proxies = None

parser = argparse.ArgumentParser(usage=description)
subparsers = parser.add_subparsers(title='valid operations', dest='COMMAND')
subparsers.required = True
submit_parser = subparsers.add_parser('submit')
submit_parser.add_argument('submit_filename', metavar='FILE', type=str, help='File for submission')
submit_parser.add_argument('--submit_comment', metavar='COMMENT', type=str, default='',
                           help='Optional comment describing this submission')


def submit(filename, user_comment):
    farch = filename + '.gz'
    with open(filename,'rb') as inp, gzip.GzipFile(farch, 'wb') as outp:
        outp.write(inp.read())
    data = dict(user_comment=user_comment, **auth_options)
    files = dict(file=(farch, open(farch, 'rb')))
    st = time()
    response = requests.post(server + '/api/submit', data=data, files=files, proxies=proxies)
    print('Uploaded in %ds' % (time()-st))
    print(response.text)


if __name__ == '__main__':
    args = parser.parse_args()
    submit(filename=args.submit_filename, user_comment=args.submit_comment)
