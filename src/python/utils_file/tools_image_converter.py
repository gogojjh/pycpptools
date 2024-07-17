# Usage: python batch_convert.py ./images jpg ./converted_images

import os
import argparse
from PIL import Image

def convert_image(input_file, output_format, output_directory):
  img = Image.open(input_file)
  file_name = os.path.basename(input_file)
  base_name, _ = os.path.splitext(file_name)
  output_file = os.path.join(output_directory, f"{base_name}.{output_format}")
  img.save(output_file)
  print(f"Converted {input_file} to {output_file}")

def batch_convert(input_directory, output_format, output_directory):
  if not os.path.exists(output_directory):
    os.makedirs(output_directory)

  for root, _, files in os.walk(input_directory):
    for file in files:
      if file.lower().endswith('.jpeg'):
        input_file = os.path.join(root, file)
        convert_image(input_file, output_format, output_directory)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Batch convert JPEG images to another format.')
  parser.add_argument('input_directory', type=str, help='Directory containing JPEG images to convert')
  parser.add_argument('output_format', type=str, choices=['jpg', 'png', 'bmp', 'gif', 'tiff'], help='Output image format')
  parser.add_argument('output_directory', type=str, help='Directory to save converted images')
  args = parser.parse_args()

  batch_convert(args.input_directory, args.output_format, args.output_directory)
