import os
from ctypes import *
import random
import jsonschema
import asyncio
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from inference.base_inference_engine import AbstractInferenceEngine
from inference.exceptions import InvalidModelConfiguration, InvalidInputData, ApplicationError


def sample(probs):
	s = sum(probs)
	probs = [a / s for a in probs]
	r = random.uniform(0, 1)
	for i in range(len(probs)):
		r = r - probs[i]
		if r <= 0:
			return i
	return len(probs) - 1


def c_array(ctype, values):
	arr = (ctype * len(values))()
	arr[:] = values
	return arr


class BOX(Structure):
	_fields_ = [("x", c_float),
				("y", c_float),
				("w", c_float),
				("h", c_float)]


class DETECTION(Structure):
	_fields_ = [("bbox", BOX),
				("classes", c_int),
				("prob", POINTER(c_float)),
				("mask", POINTER(c_float)),
				("objectness", c_float),
				("sort_class", c_int),
				("uc", POINTER(c_float)),
				("points", c_int)]


class IMAGE(Structure):
	_fields_ = [("w", c_int),
				("h", c_int),
				("c", c_int),
				("data", POINTER(c_float))]


class METADATA(Structure):
	_fields_ = [("classes", c_int),
				("names", POINTER(c_char_p))]


class InferenceEngine(AbstractInferenceEngine):

	def __init__(self, model_path):
		self.net = None
		# image width scale
		self.resize_w = 0
		# image height scale
		self.resize_h = 0
		# mean average precision
		self.pmap = None
		# relative size
		self.relative = 0
		# image aspect ratio
		self.letter = 0
		self.lib = CDLL("/main/darknet/libdarknet.so", RTLD_GLOBAL)
		self.lib.network_width.argtypes = [c_void_p]
		self.lib.network_width.restype = c_int
		self.lib.network_height.argtypes = [c_void_p]
		self.lib.network_height.restype = c_int

		self.predict = self.lib.network_predict
		self.predict.argtypes = [c_void_p, POINTER(c_float)]
		self.predict.restype = POINTER(c_float)
		self.set_gpu = self.lib.cuda_set_device
		self.set_gpu.argtypes = [c_int]
		self.init_cpu = self.lib.init_cpu
		self.make_image = self.lib.make_image
		self.make_image.argtypes = [c_int, c_int, c_int]
		self.make_image.restype = IMAGE

		self.get_network_boxes = self.lib.get_network_boxes
		self.get_network_boxes.argtypes = [
			c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
		self.get_network_boxes.restype = POINTER(DETECTION)

		self.make_network_boxes = self.lib.make_network_boxes
		self.make_network_boxes.argtypes = [c_void_p]
		self.make_network_boxes.restype = POINTER(DETECTION)

		self.free_detections = self.lib.free_detections
		self.free_detections.argtypes = [POINTER(DETECTION), c_int]

		self.free_ptrs = self.lib.free_ptrs
		self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

		self.network_predict = self.lib.network_predict_ptr
		self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

		self.reset_rnn = self.lib.reset_rnn
		self.reset_rnn.argtypes = [c_void_p]

		self.load_net = self.lib.load_network
		self.load_net.argtypes = [c_char_p, c_char_p, c_int]
		self.load_net.restype = c_void_p

		self.load_net_custom = self.lib.load_network_custom
		self.load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
		self.load_net_custom.restype = c_void_p

		self.do_nms_obj = self.lib.do_nms_obj
		self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

		self.do_nms_sort = self.lib.do_nms_sort
		self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

		self.free_image = self.lib.free_image
		self.free_image.argtypes = [IMAGE]

		self.letterbox_image = self.lib.letterbox_image
		self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
		self.letterbox_image.restype = IMAGE

		self.load_meta = self.lib.get_metadata
		self.lib.get_metadata.argtypes = [c_char_p]
		self.lib.get_metadata.restype = METADATA

		self.load_image = self.lib.load_image_color
		self.load_image.argtypes = [c_char_p, c_int, c_int]
		self.load_image.restype = IMAGE

		self.rgbgr_image = self.lib.rgbgr_image
		self.rgbgr_image.argtypes = [IMAGE]

		self.predict_image = self.lib.network_predict_image
		self.predict_image.argtypes = [c_void_p, IMAGE]
		self.predict_image.restype = POINTER(c_float)
		self.predict_image_letterbox = self.lib.network_predict_image_letterbox
		self.predict_image_letterbox.argtypes = [c_void_p, IMAGE]
		self.predict_image_letterbox.restype = POINTER(c_float)
		self.meta = self.load_meta(self.__ensure__ctype__string(os.path.join(model_path, 'obj.data')))
		self.font = ImageFont.truetype("/main/fonts/DejaVuSans.ttf", 20)

		super().__init__(model_path)

	def load(self):
		with open(os.path.join(self.model_path, 'config.json')) as f:
			data = json.load(f)
		try:
			self.validate_json_configuration(data)
			self.set_model_configuration(data)
		except ApplicationError as e:
			raise e

		with open(os.path.join(self.model_path, 'obj.names'), 'r') as f:
			self.labels = [line.strip() for line in f.readlines()]
		self.net = self.load_net(self.__ensure__ctype__string(os.path.join(self.model_path, 'yolo-obj.cfg')),
								 self.__ensure__ctype__string(os.path.join(self.model_path, 'yolo-obj.weights')), 0)

	def __ensure__ctype__string(self, string):
		return bytes(string, encoding='utf-8')

	def array_to_image(self, arr):
		arr = arr.transpose(2, 0, 1)
		c = arr.shape[0]
		h = arr.shape[1]
		w = arr.shape[2]
		arr = (arr / 255.0).flatten()
		data = c_array(c_float, arr)
		im = IMAGE(w, h, c, data)
		return im

	async def infer(self, input_data, draw, predict_batch):
		await asyncio.sleep(0.00001)
		try:
			pillow_image = Image.open(input_data.file).convert('RGB')
			np_image = np.array(pillow_image)
		except Exception as e:
			raise InvalidInputData('corrupted image')
		try:
			with open(self.model_path + '/config.json') as f:
				data = json.load(f)
		except Exception as e:
			raise InvalidModelConfiguration('config.json not found or corrupted')

		detection_threshold = data['detection_threshold']
		non_max_suppression_threshold = data['nms_threshold']
		hier_threshold = data['hier_threshold']
		height, width, depth = np_image.shape
		num = c_int(0)
		pnum = pointer(num)
		im = self.array_to_image(np_image)
		self.predict_image(self.net, im)
		dets = self.get_network_boxes(self.net, width, height, detection_threshold, hier_threshold, self.pmap, self.relative,
									  pnum, self.letter)
		num = pnum[0]
		self.do_nms_obj(dets, num, self.meta.classes, non_max_suppression_threshold)
		res = []
		for j in range(num):
			for i in range(self.meta.classes):
				if dets[j].prob[i] > detection_threshold:
					b = dets[j].bbox
					left = (b.x - b.w / 2) 

					right = (b.x + b.w / 2)

					top = (b.y - b.h / 2) 

					bottom = (b.y + b.h / 2)



					if (left < 0):
						left = 0
					if (right > im.w - 1):
						right = im.w - 1
					if (top < 0):
						top = 0
					if (bottom > im.h - 1):
						bottom = im.h - 1
					

					res.append(
						{
							'ObjectClassId': i,
							'ObjectClassName': self.meta.names[i].decode('utf-8'),
							'confidence': dets[j].prob[i] * 100,
							'coordinates': {
								'left': left,
								'top': top,
								'right': right,
								'bottom': bottom
							}
						}
					)
		self.free_detections(dets, num)
		if predict_batch:
			response = dict([('bounding-boxes', res), ('ImageName', input_data.filename)])
		else:
			response = dict([('bounding-boxes', res)])
		if not draw:
			return response
		else:
			try:
				self.draw_image(pillow_image, response)
			except ApplicationError as e:
				raise e
			except Exception as e:
				raise e

	async def run_batch(self, input_data, draw, predict_batch):
		result_list = []
		for image in input_data:
			post_process = await self.infer(image, draw, predict_batch)
			if post_process is not None:
				result_list.append(post_process)
		return result_list

	def draw_image(self, image, response):
		"""
		Draws on image and saves it.
		:param image: image of type pillow image
		:param response: inference response
		:return:
		"""
		draw = ImageDraw.Draw(image)
		for bbox in response['bounding-boxes']:
			draw.rectangle([bbox['coordinates']['left'], bbox['coordinates']['top'], bbox['coordinates']['right'],
							bbox['coordinates']['bottom']], outline="red")
			left = bbox['coordinates']['left']
			top = bbox['coordinates']['top']
			conf = "{0:.2f}".format(bbox['confidence'])
			draw.text((int(left), int(top) - 20), str(conf) + "% " + str(bbox['ObjectClassName']), 'red', self.font)
		image.save('/main/result.jpg', 'PNG')

	def free(self):
		pass

	def validate_configuration(self):
		# check if network architecture file exists
		if not os.path.exists(os.path.join(self.model_path, 'yolo-obj.cfg')):
			raise InvalidModelConfiguration('yolo-obj.cfg not found')
		# check if weights file exists
		if not os.path.exists(os.path.join(self.model_path, 'yolo-obj.weights')):
			raise InvalidModelConfiguration('yolo-obj.weights not found')
		# check if labels file exists
		if not os.path.exists(os.path.join(self.model_path, 'obj.names')):
			raise InvalidModelConfiguration('obj.names not found')
		# check if data file exists
		if not os.path.exists(os.path.join(self.model_path, 'obj.data')):
			raise InvalidModelConfiguration('obj.data not found')
		return True

	def set_model_configuration(self, data):
		self.configuration['framework'] = data['framework']
		self.configuration['type'] = data['type']
		self.configuration['network'] = data['network']

	def validate_json_configuration(self, data):
		with open(os.path.join('inference', 'ConfigurationSchema.json')) as f:
			schema = json.load(f)
		try:
			jsonschema.validate(data, schema)
		except Exception as e:
			raise InvalidModelConfiguration(e)
