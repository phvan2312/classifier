import argparse
import os
from shutil import rmtree # for remove folder
from distutils.dir_util import copy_tree # for copy
from shutil import copyfile

must_have_folder = ['data/preprocess','nn']
must_have_files  = ['at_classifier.py','at_classifier_client.py','at_classifier_server.py','train.py','train_continue.py','train_hyper.py',
                    'utils.py','vars.py']

def copy(src_folder_path, dest_folder_path):
    if os.path.isfile(dest_folder_path):
        rmtree(dest_folder_path)

    copy_tree(src_folder_path, dest_folder_path)

def move(new_path,saved_model_name):
    assert os.path.exists(new_path)

    # copy necessary folders
    prefix_folder = './'
    for folder in must_have_folder:
        from_path = prefix_folder + folder
        to_path = os.path.join(new_path,folder)

        assert os.path.exists(from_path)

        copy(src_folder_path=from_path,dest_folder_path=to_path)

    # copy necessary files
    prefix_file = './'
    for file in must_have_files:
        from_path = prefix_file + file
        to_path = os.path.join(new_path,file)

        assert os.path.exists(from_path)

        copyfile(from_path, to_path)

    # copy saved_model path
    copy(src_folder_path=os.path.join('./','saved_model',saved_model_name),
         dest_folder_path=os.path.join(new_path,'saved_model',saved_model_name))

def main():
    move(new_path=args.new_path,saved_model_name=args.saved_model_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="script to create Templae automatically",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--new_path', type=str, required=True, help='new folder path')
    parser.add_argument('--saved_model_name', type=str, required=True, help='model name')

    args = parser.parse_args()
    main()