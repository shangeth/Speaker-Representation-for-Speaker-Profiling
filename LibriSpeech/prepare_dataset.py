import os
import shutil
import argparse

my_parser = argparse.ArgumentParser(description='Path to the LibriSpeech dataset folder')
my_parser.add_argument('--path',
                       metavar='path',
                       type=str,
                       help='the path to dataset folder')
args = my_parser.parse_args()


original_data_path = os.path.join(args.path, 'dev-clean')
# 'train-clean-360') # cange if other folders(like dev-clean) are used
final_dir = os.path.join(args.path, 'final_repr_data')

if not os.path.exists(final_dir): 
    os.mkdir(final_dir)

for speaker_id in os.listdir(original_data_path):
    for uttid in os.listdir(os.path.join(original_data_path, speaker_id)):
        for file in os.listdir(os.path.join(original_data_path, speaker_id, uttid)):
            if file.endswith('.flac'):
                src = os.path.join(original_data_path, speaker_id, uttid, file)
                if not os.path.exists(os.path.join(final_dir, speaker_id)): 
                    os.mkdir(os.path.join(final_dir, speaker_id))
                dst = os.path.join(final_dir, speaker_id, file)
                shutil.copy(src, dst)

print('Final Path to LibriSpeech dataset = ', final_dir)