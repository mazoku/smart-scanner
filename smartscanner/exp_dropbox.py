import sys

import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError


def load_token(fname='../data/.dbx_access_token'):
    with open(fname, 'r') as f:
        token = f.read()
    return token


def establish_connection(token):
    # Create an instance of a Dropbox class, which can make requests to the API.
    print 'Creating a Dropbox object ...',
    dbx = dropbox.Dropbox(token)
    print 'done'

    # Check that the access token is valid
    try:
        dbx.users_get_current_account()
    except AuthError as err:
        sys.exit("ERROR: Invalid access token; try re-generating an "
                 "access token from the app console on the web.")

    return dbx


def list_content(dbx, path=''):
    print 'Accessing content ...'
    # list all of the content in the root directory
    entries = sorted(dbx.files_list_folder(path).entries, key=lambda x: x.name.lower())
    for e in entries:
        print e.name


def upload_file(dbx, local_fname, dbx_fname):
    # upload a file
    # with open(local_fname, 'r') as f:
    #     print 'Uploading {} to Dropbox as {}...'.format(local_fname, dbx_fname)
    #     try:
    #         dbx.files_upload(f.read(), dbx_fname, mode=WriteMode('overwrite'))
    #     except ApiError as err:
    #         if err.error.is_path() and err.error.get_path().error.is_insufficient_space():
    #             sys.exit('ERROR: Cannot upload the file - insufficient space.')
    #         elif err.user_message_text:
    #             print err.user_message_text
    #             sys.exit()
    #         else:
    #             print err
    #             sys.exit()

    # upload file (image)
    with open(local_fname, 'rb') as f:
        try:
            dbx.files_upload(f.read(), dbx_fname, mode=WriteMode('overwrite'))
        except Exception as err:
            print 'Failed to upload {}: {}'.format(local_fname, err)


if __name__ == '__main__':
    # load access token
    token = load_token()

    # create dbx object
    dbx = establish_connection(token)

    # list content
    # list_content(dbx, path='')

    # upload file
    # fname = 'test_file.txt'
    # dbx_fname = '/scans/SmartScanner/' + fname
    # with open(fname, 'w') as f:
    #     f.write('This is python-dropbox test.')
    fname = '../data/eye_scan.png'
    dbx_fname = '/scans/SmartScanner/' + fname.split('/')[-1]
    upload_file(dbx, fname, dbx_fname)