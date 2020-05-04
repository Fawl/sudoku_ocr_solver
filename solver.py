import torch
import numpy as np
import cv2
from PIL import Image
import sys
import time

from train import classifier

### OCR ###

def load_image(filename, output_type='string'):
	img = Image.open(filename).convert('LA')

	brightness_array = np.array(img)[:, :, 0]
	brightness_array = 255-brightness_array
	divisor = brightness_array.shape[0] // 9

	puzzle = list()

	for i in range(9):
		row = list()
		for j in range(9):
			row.append(cv2.resize(brightness_array[i*divisor:(i+1)*divisor, j*divisor:(j+1)*divisor][3:-3, 3:-3], dsize=(28, 28), interpolation=cv2.INTER_CUBIC))
		puzzle.append(row)

	# plt.imshow(puzzle[4][4], cmap='gray')
	# plt.show()

	model = classifier()
	model.load_state_dict(torch.load('saved_models/mnist_cnn.pt'))
	model.eval()

	numbers = ''

	for i, row in enumerate(puzzle):
		for j, image in enumerate(row):
			if np.mean(image) > 6:
				norm_image = torch.from_numpy(image.reshape(1, 1, 28, 28).astype('float32')/255)
				out = model(norm_image)
				prediction = out.argmax(dim=1, keepdim=True).item()
				numbers += str(prediction)
			else:
				numbers+='0'

	if output_type == 'string':
		return numbers

	elif output_type == 'array':
		return np.array([int(num) for num in list(numbers)]).reshape((9, 9))


### Solver Logic ###

def same_row(i,j): return (i/9 == j/9)
def same_col(i,j): return (i-j) % 9 == 0
def same_block(i,j): return (i/27 == j/27 and i%9/3 == j%9/3)

def r(a):
	i = a.find('0')
	if i == -1:
		a_list = [int(num) for num in list(a)]
		a_array = np.array(a_list).reshape((9, 9))
		print(a_array)
		print(f'\nSolved in {round(time.time()-start, 3)} seconds!')
		sys.exit()

	excluded_numbers = set()
	for j in range(81):
		if same_row(i,j) or same_col(i,j) or same_block(i,j):
			excluded_numbers.add(a[j])

	for m in '123456789':
		if m not in excluded_numbers:
			r(a[:i]+m+a[i+1:])

	

if __name__ == '__main__':
	start = time.time()
	for_display = load_image('sample.png', output_type='array')
	print('--- Input Puzzle ---')
	print(for_display)
	numbers = load_image('sample.png')
	print('\n--- Solution ---')
	r(numbers)