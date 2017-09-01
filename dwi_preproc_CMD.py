#! /usr/bin/env python
import sys
import optparse

def main():
    p = optparse.OptionParser()
    p.add_option('--subject_list')
    p.add_option('--base_directory')
    p.add_option('--out_directory')
    options, arguments = p.parse_args()
    subject_list = options.subject_list
    base_directory = options.base_directory
    out_directory = options.out_directory

    try:
        from dwi_workflows import dwi_preproc
        dwi_preproc(subject_list,base_directory,out_directory)
    except:
        print 'ERROR'
    
if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())   
