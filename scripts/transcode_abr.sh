#!/usr/bin/env bash
# Multi-rung HLS + DASH-friendly renditions with FFmpeg (adjust paths and bitrates to match ladder in backend).
set -euo pipefail
INPUT="${1:?usage: transcode_abr.sh input.mp4 [out_dir]}"
OUT="${2:-./media/out}"
mkdir -p "$OUT"

ffmpeg -y -i "$INPUT" -filter_complex "[0:v]split=4[v1][v2][v3][v4]; \
[v1]scale=-2:1080[v1080]; [v2]scale=-2:720[v720]; [v3]scale=-2:480[v480]; [v4]scale=-2:360[v360]" \
  -map "[v1080]" -c:v:0 libx264 -b:v:0 8M -maxrate:v:0 9M -bufsize:v:0 16M -g 48 -keyint_min 48 \
  -map "[v720]" -c:v:1 libx264 -b:v:1 5M -maxrate:v:1 5.5M -bufsize:v:1 10M -g 48 -keyint_min 48 \
  -map "[v480]" -c:v:2 libx264 -b:v:2 2.5M -maxrate:v:2 2.8M -bufsize:v:2 5M -g 48 -keyint_min 48 \
  -map "[v360]" -c:v:3 libx264 -b:v:3 1.2M -maxrate:v:3 1.4M -bufsize:v:3 2.5M -g 48 -keyint_min 48 \
  -map a:0 -c:a aac -b:a 128k -ac 2 \
  -f hls -hls_time 4 -hls_playlist_type vod -hls_flags independent_segments \
  -hls_segment_filename "$OUT/stream_%v/data%03d.ts" \
  -master_pl_name master.m3u8 \
  -var_stream_map "v:0,a:0 v:1,a:0 v:2,a:0 v:3,a:0" \
  "$OUT/stream_%v/playlist.m3u8"

echo "HLS master: $OUT/master.m3u8 — point your packager (e.g. shaka / ffmpeg DASH) at these renditions for DASH."
