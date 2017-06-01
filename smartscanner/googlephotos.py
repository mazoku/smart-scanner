#!/usr/bin/python
import gdata.photos.service
import os.path
import sys, os

#Script to find and download the oversized photos in Google Plus

programName = 'SmartScanner'
# username = "<Your Email>"
# password = "<Your password>"
# albumName = "Album ID"
# album_url = 'https://goo.gl/photos/GxWrWDDCf4S8cSfZA'
username = raw_input("Username: ")
password = raw_input("Password: ")
albumName = 'smartscanner'

#Google APIs only allow scanning 1000 photos at a time, so one has to increase this value every time 1, 1000, 2000 etc
offset = 1
outDir = "supersize"

# If any photos is greater than 2048 pixel in either side , it will add to the quota
maxWidth = 2048
maxHeight = 2048

# Authenticate to Picasa Web Albums.
gd_client = gdata.photos.service.PhotosService()
gd_client.email = username
gd_client.password = password
gd_client.source = programName
gd_client.ProgrammaticLogin()

# imgmax=d is important to download the full res photos
photos = gd_client.GetFeed(
    '/data/feed/api/user/%s/albumid/%s?kind=photo&imgmax=d&max-results=1000&start-index=%s' % (
        username, albumName, offset))
# count=0
# size=0
print "Found", len(photos.entry), " photos in the album"
for photo in photos.entry:
    # if (int(photo.width.text) > maxWidth) or (int(photo.height.text) > maxHeight):
    pname = photo.title.text
    print "PHOTO:", photo.title.text, int(photo.width), " X ", int(photo.height), "SIZE: ", int(photo.size.text)/1000000, " Mb"
    # url = photo.GetMediaURL()
    # media = gd_client.GetMedia(url)

    # print media.file_name, media.content_type, media.content_length
    # data = media.file_handle.read()
    # media.file_handle.close()

    # filename = "%s/%s" % (outDir, pname)
    # print "Output: %s" % filename
    # sys.stdout.flush()

    # if not os.path.isdir(outDir):
    #     os.mkdir(outDir)

    # out = open(filename, 'wb')
    # out.write(data)
    # out.close()

    # count +=1
    # size += int(photo.size.text)
# print "Total big photos ",count, " taking space ", size / 1000000, " Mb"