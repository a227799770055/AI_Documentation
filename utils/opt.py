import argparse


def video_parse_opt():
    parser = argparse.ArgumentParser(
        prog="inference_video.py",
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='model/classified_1012.pth',
        help='path of the config.'
    )
    parser.add_argument(
        '--video_path',
        type=str,
        default='video/anal.mp4',
        help='path of the video.'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='output',
        help='path of the result.'  
    )
    opt = parser.parse_args()
    return opt