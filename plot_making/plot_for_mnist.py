import os
import subprocess
import glob

_16_bit_files = sorted(glob.glob('./mnist_stuff/*h16*'))
_24_bit_files = sorted(glob.glob('./mnist_stuff/*h24*'))
_32_bit_files = sorted(glob.glob('./mnist_stuff/*h32*'))
_48_bit_files = sorted(glob.glob('./mnist_stuff/*h48*'))

def run_file_set(file_set,outFile):
    args = ['python3','plot_maker_ours_only.py',outFile]
    for inFile in file_set:
        args.append(inFile)
        if 'accv' in inFile.lower():
            args.append('Likelihood Loss')
        if 'triplet' in inFile.lower():
            args.append('Margin Loss')
        if 'simple' in inFile.lower():
            args.append('Simple Spring Loss')
        if 'spring' in inFile.lower() and not 'simple' in inFile.lower():
            args.append('Spring Loss')
    subprocess.run(args)


run_file_set(_16_bit_files,'mnist_16bit.eps')
run_file_set(_24_bit_files,'mnist_24bit.eps')
run_file_set(_32_bit_files,'mnist_32bit.eps')
run_file_set(_48_bit_files,'mnist_64bit.eps')


