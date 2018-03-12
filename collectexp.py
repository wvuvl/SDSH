from os import listdir
from os.path import isdir, join
import os 
import re


def main():
    parsesched = re.compile('exp_(\w+)')
    exps = [f for f in listdir(os.getcwd())]

    for s in exps:
        exp = parsesched.findall(s)
        
        if len(exp) == 0:
            continue
            
        expname = exp[0]
        print("\n%s\n%s\n%s" % ("=" * 80, expname, "=" * 80))
        folders = [f for f in listdir(s) if isdir(join(s, f))]
        parsename = re.compile('^([\w|\.]+)_h(\d+)_([\w|\.|\-|\d]+)$')
        resultparse = re.compile('(RandomSearch|SITQ|ITQ|SSH|No rotation): Test on train: \d+.\d+; Test on test: (\d+.\d+)')

        exp_dict = {}

        for d in folders:
            x = parsename.findall(d)[0]
            name = x[0]
            hash_size = int(x[1])
            details = x[2]
            has_random = False
            has_sitq = False
            has_itq = False
            has_ssh = False
            has_no_r = False
            if name not in exp_dict:
                exp_dict[name] = {}
            exp_dict[name][hash_size] = {}
            try:
                for line in reversed(list(open(join(s, d, 'results.txt')))):
                    r = resultparse.findall(line.rstrip())
                    if len(r) == 0:
                        break
                    r = r[0]
                    if has_random and has_sitq and has_itq and has_ssh and has_no_r:
                        break
                    if r[0] == 'RandomSearch':
                        has_random = True
                    if r[0] == 'SITQ':
                        has_sitq = True
                    if r[0] == 'ITQ':
                        has_itq = True
                    if r[0] == 'SSH':
                        has_ssh = True
                    if r[0] == 'No rotation':
                        has_no_r = True
                    exp_dict[name][hash_size][r[0]] = r[1]
            except:
                print("no results.txt")

        for type in ['RandomSearch', 'SITQ', 'ITQ', 'SSH', 'No rotation']:
            print("\n%s:" % type)
            for name in exp_dict:
                l = []
                for hs in exp_dict[name]:
                    l.append(hs)
                l = sorted(l)
                for hs in l:
                    if type in exp_dict[name][hs]:
                        print("%s, %2d, %s" % (name, hs, exp_dict[name][hs][type]))

if __name__ == '__main__':
    main()
