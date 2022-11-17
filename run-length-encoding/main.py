from run_length_encoding import RunLengthEncoding

for video in ["video1", "video2", "video3"]:
    video_path = "./../src/videos" + video + ".mp4"

    # create an instance of RLE
    rle = RunLengthEncoding(video_path)
    rle.run()