#!/usr/bin/python
# coding: utf8

import argparse
import hashlib
import io
import mimetypes
import os.path
import re
from datetime import datetime

from apiclient import http
# from apiclient.http import MediaFileUpload
from appdirs import AppDirs
# from progressbar import (AdaptiveETA, AdaptiveTransferSpeed, Bar, Percentage,
#                          ProgressBar)
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

APPNAME = "smartscanner"

###############################################################################
# Useful classes
###############################################################################


class NoGooglePhotosFolderError(Exception):
    pass


class LocalMedia(object):

    CHUNK_SIZE = 4096

    def __init__(self, media_path):
        self.path = media_path

    @property
    def filename(self):
        return os.path.basename(self.path)

    @property
    def canonical_filename(self):
        return self.filename

    @property
    def size(self):
        return os.path.getsize(self.path)

    @property
    def checksum(self):
        md5sum = hashlib.md5()
        with io.open(self.path, 'rb') as media_file:
            chunk_reader = lambda: media_file.read(LocalMedia.CHUNK_SIZE)
            for chunk in iter(chunk_reader, b""):
                md5sum.update(chunk)

        return md5sum.hexdigest()

    @property
    def mimetype(self):
        mimetype, _ = mimetypes.guess_type(self.path)
        return mimetype


class GooglePhotosMedia(object):

    SPECIAL_SUFFIXES_PATTERN = r"(?:-EFFECTS|-ANIMATION|-COLLAGE|-PANO|~\d+)$"
    CANONICAL_FILENAME_FORMAT = (
        "%(date)s-%(camera_model)s%(camera_owner)s%(suffix)s.%(extension)s")

    def __init__(self, drive_file):
        self.drive_file = drive_file

    def get_custom_property_value(self, key):
        for prop in self.drive_file["properties"]:
            if prop["key"] == key:
                return prop["value"]

        raise KeyError()

    def get_exif_value(self, tag_name):
        try:
            exif_override_property_name = "exif-%s" % tag_name
            return self.get_custom_property_value(exif_override_property_name)
        except KeyError:
            return self.drive_file["imageMediaMetadata"][tag_name]

    @property
    def date(self):
        try:
            exif_date = self.get_exif_value("date")
            photo_date = datetime.strptime(exif_date, "%Y:%m:%d %H:%M:%S")
        except (KeyError, ValueError):
            import_date = self.drive_file["createdDate"]
            photo_date = datetime.strptime(import_date,
                                           "%Y-%m-%dt%H:%M:%S.000Z")

        return photo_date

    @property
    def size(self):
        return int(self.drive_file["fileSize"])

    @property
    def checksum(self):
        return self.drive_file["md5Checksum"]

    @property
    def id(self):
        return self.drive_file["id"]

    @property
    def camera_owner(self):
        try:
            artist = self.get_exif_value("artist")
            match = re.match("Camera Owner, ([^;]+)(?:;|$)", artist)
            camera_owner = match.group(1) if match else artist
        except KeyError:
            camera_owner = None

        return camera_owner

    @property
    def camera_model(self):
        try:
            camera_model = self.get_exif_value("cameraModel")
        except KeyError:
            if re.match(r"IMG-\d{8}-WA\d+", self.filename):
                camera_model = "WhatsApp"
            else:
                camera_model = None

        return camera_model

    @property
    def extension(self):
        return self.drive_file["fileExtension"]

    @property
    def filename(self):
        return self.drive_file["title"]

    @property
    def suffix(self):
        filename, _ = os.path.splitext(self.drive_file["title"])
        suffix_match = re.search(GooglePhotosMedia.SPECIAL_SUFFIXES_PATTERN,
                                 filename, re.VERBOSE)
        suffix = suffix_match.group(0) if suffix_match else None
        return suffix

    @property
    def canonical_filename(self):
        def sanitize_name(name):
            return re.sub(r"[^\w]", "", name)

        camera_model = self.camera_model or "Unknown"
        camera_owner = self.camera_owner or ""
        suffix = self.suffix or ""

        if self.camera_owner:
            camera_owner = "-" + sanitize_name(camera_owner)

        canonical_filename = GooglePhotosMedia.CANONICAL_FILENAME_FORMAT % {
            "date": self.date.strftime("%Y%m%d-%H%M%S"),
            "camera_model": sanitize_name(camera_model),
            "camera_owner": camera_owner,
            "suffix": suffix,
            "extension": self.extension.lower(),
        }

        return canonical_filename


class GooglePhotosSync(object):

    GOOGLE_PHOTO_FOLDER_QUERY = (
        'title = "Google Photos" and "root" in parents and trashed=false')
    MEDIA_QUERY = '"%s" in parents and trashed=false'
    PAGE_SIZE = 100

    def __init__(self,
                 target_folder=".",
                 client_secret_file="client_secret.json",
                 credentials_file="credentials.json"):

        self.target_folder = target_folder
        self.gauth = GoogleAuth()
        self.gauth.settings["client_config_file"] = client_secret_file
        self.gauth.settings["save_credentials_file"] = credentials_file
        self.gauth.settings["save_credentials"] = True
        self.gauth.settings["save_credentials_backend"] = "file"
        self.gauth.settings["get_refresh_token"] = True
        self.gauth.CommandLineAuth()
        self.googleDrive = GoogleDrive(self.gauth)

    def _get_photos_folder_id(self):
        query_results = self.googleDrive.ListFile(
            {"q": GooglePhotosSync.GOOGLE_PHOTO_FOLDER_QUERY}).GetList()
        try:
            return query_results[0]["id"]
        except:
            raise NoGooglePhotosFolderError()

    def get_remote_medias(self):
        googlePhotosFolderId = self._get_photos_folder_id()
        query_params = {
            "q": GooglePhotosSync.MEDIA_QUERY % googlePhotosFolderId,
            "maxResults": GooglePhotosSync.PAGE_SIZE
        }
        for page_results in self.googleDrive.ListFile(query_params):
            for drive_file in page_results:
                if drive_file["mimeType"].startswith("video/"):
                    continue
                yield GooglePhotosMedia(drive_file)

    def get_remote_media_by_name(self, filename):
        googlePhotosFolderId = self._get_photos_folder_id()
        query_params = {
            "q": 'title = "%s" and "%s" in parents and trashed=false' %
            (filename, googlePhotosFolderId)
        }
        found_media = self.googleDrive.ListFile(query_params).GetList()
        return GooglePhotosMedia(found_media[0]) if found_media else None

    def get_local_medias(self):
        for directory, _, files in os.walk(self.target_folder):
            for filename in files:
                media_path = os.path.join(directory, filename)
                mimetype, _ = mimetypes.guess_type(media_path)
                if mimetype and mimetype.startswith('image/'):
                    yield LocalMedia(media_path)

    def get_target_folder(self, media):
        year_month_folder = media.date.strftime("%Y/%m")
        target_folder = os.path.join(self.target_folder, year_month_folder)
        return target_folder

    def has_local_version(self, media):
        target_folder = self.get_target_folder(media)
        local_filename = os.path.join(target_folder, media.canonical_filename)
        return os.path.isfile(local_filename)

    def downloadMedia(self, media, progress_handler=None):
        target_folder = self.get_target_folder(media)
        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)

        target_filename = os.path.join(target_folder, media.canonical_filename)

        with io.open(target_filename, 'bw') as target_file:
            request = self.gauth.service.files().get_media(fileId=media.id)
            download_request = http.MediaIoBaseDownload(target_file, request)

            done = False
            while not done:
                download_status, done = download_request.next_chunk()
                if progress_handler is not None:
                    progress_handler.update_progress(download_status)

        return target_filename

    # def uploadMedia(self, local_media, progress_handler=None):
    #
    #     remote_media = self.get_remote_media_by_name(local_media.filename)
    #
    #     media_body = MediaFileUpload(local_media.path, resumable=True)
    #
    #     if remote_media:
    #         upload_request = self.gauth.service.files().update(
    #             fileId=remote_media.id,
    #             body=remote_media.drive_file,
    #             newRevision=True,
    #             media_body=media_body)
    #     else:
    #         body = {
    #             'title': local_media.filename,
    #             'mimetype': local_media.mimetype
    #         }
    #         upload_request = self.gauth.service.files().insert(
    #             body=body,
    #             media_body=media_body)
    #
    #     done = False
    #     while not done:
    #         upload_status, done = upload_request.next_chunk()
    #         if progress_handler is not None:
    #             progress_handler.update_progress(upload_status)


# class ProgressHandler(object):
#     def __init__(self, media):
#
#         progress_bar_widgets = [
#             media.canonical_filename,
#             "    ",
#             Percentage(),
#             Bar(),
#             " ",
#             AdaptiveTransferSpeed(),
#             "  ",
#             AdaptiveETA(),
#         ]
#
#         self.progress_bar = ProgressBar(maxval=media.size,
#                                         term_width=160,
#                                         widgets=progress_bar_widgets)
#         self.progress_bar.start()
#
#     def update_progress(self, status):
#         if not status or status.progress() == 1:
#             self.progress_bar.finish()
#
#         else:
#             size_downloaded = status.progress() * status.total_size
#             self.progress_bar.update(size_downloaded)

###############################################################################
# Command functions
###############################################################################


def download_command(googlePhotosSync):

    progress_handler = None
    for remote_media in googlePhotosSync.get_remote_medias():

        if remote_media.extension == "gif":
            continue

        if googlePhotosSync.has_local_version(remote_media):
            continue

        # if args.dry_run:
        #     print "Downloading %s" % remote_media.canonical_filename
        #     continue
        #
        # if not args.quiet:
        #     progress_handler = ProgressHandler(remote_media)

        googlePhotosSync.downloadMedia(remote_media, progress_handler=progress_handler)


# def re_upload_command(googlePhotosSync, args):
#
#     progress_handler = None
#     for local_media in googlePhotosSync.get_local_medias():
#
#         remote_media = googlePhotosSync.get_remote_media_by_name(
#             local_media.filename)
#
#         if not remote_media or remote_media.checksum == local_media.checksum:
#             continue
#
#         if args.dry_run:
#             print "Re-uploading %s" % remote_media.canonical_filename
#             continue
#
#         if not args.quiet:
#             progress_handler = ProgressHandler(local_media)
#
#         googlePhotosSync.uploadMedia(local_media, progress_handler=progress_handler)

###############################################################################
# Main code
###############################################################################

# parser = argparse.ArgumentParser(
#     description="Google Photos simple synchronization tool")
# parser.add_argument("--quiet", help="quiet (no output)")
# parser.add_argument(
#     "--dry-run",
#     action='store_true',
#     help="show what would have been transfered")
# parser.add_argument(
#     "command",
#     metavar="COMMAND",
#     choices=["re-upload", "download"],
#     help="command to execute")
# parser.add_argument(
#     "target_folder",
#     metavar="TARGET_FOLDER",
#     help="The photos will be transfered from/to that directory")
# args = parser.parse_args()

target_folder = '/home/tomas/Dropbox/Scans/SmartScanner'

appdirs = AppDirs(APPNAME)

credentials_file = os.path.join(appdirs.user_data_dir, "credentials.json")
secret_file = os.path.join(appdirs.user_config_dir, "client_secret.json")

googlePhotosSync = GooglePhotosSync(target_folder=target_folder,
                                    client_secret_file=secret_file,
                                    credentials_file=credentials_file)

# if args.command == "download":
download_command(googlePhotosSync)

# elif args.command == "re-upload":
# re_upload_command(googlePhotosSync, args)

# vim: autoindent tabstop=4 shiftwidth=4 expandtab softtabstop=4