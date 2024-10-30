import os,glob
import zipfile


if __name__=='__main__':
    ############## SET Configurations Here ##############
    dataroot = '/data2/sample_data/ScanNet'
    DELETE_ZIP_FILE = True
    #####################################################

    sequence_zips = glob.glob(os.path.join(dataroot, 'scans', '*.zip'))
    
    for sequence_zip in sequence_zips:
        sequence_name = os.path.basename(sequence_zip).split('.')[0]
        sequence_folder = os.path.join(dataroot, 'scans', sequence_name)
        if os.path.exists(sequence_folder):
            print('[Warning] Skip {} because it already exists.'.format(sequence_folder))
            continue        
        print('------- {} -------'.format(sequence_name))
        
        # unzip the sequence
        with zipfile.ZipFile(sequence_zip, 'r') as z:
            z.extractall(os.path.join(dataroot, 'scans'))
        
        if DELETE_ZIP_FILE: # after unzipping, delete the zip file
            os.remove(sequence_zip)
            print('deleted {}'.format(sequence_zip))
        
    