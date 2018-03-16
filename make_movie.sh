#!/bin/sh

DATE=`date +%y%m%d%H%M`
MOVIECMD=`ffmpeg -r 24 -f image2 -s 1280x720 -i ./imgs/frame%d.tiff -vcodec libx264 -crf 30 test$DATE.mp4`

echo "$MOVIECMD"
